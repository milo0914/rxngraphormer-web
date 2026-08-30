#!/usr/bin/env python3
"""
Evaluate RXNGraphormer on a sample of test data with proper RDKit validation.
For retrosynthesis: each precursor set may contain multiple SMILES joined by '.'
- Split and validate EACH fragment
- A prediction is valid only if ALL fragments are valid
- Report valid prediction rate and accuracy on valid predictions
"""
import json
import torch
import numpy as np
from rdkit import Chem
from box import Box
import sys
import pickle

sys.path.insert(0, '/run/csi/mount-root/nas/4079184d856ecc166ed19d4887083405/workspaces/default/rxngraphormer-web/rxngraphormer')

from rxngraphormer.model import RXNGraphormer
from rxngraphormer.data import load_vocab, RXNG2SDataset
from rxngraphormer.utils import canonical_smiles, update_dict_key
from torch_geometric.loader import DataLoader

device = torch.device('cpu')

def validate_smiles(smi, is_retro=False):
    """Validate SMILES with RDKit. For retro, split by '.' and validate each fragment."""
    if is_retro:
        fragments = smi.split('.')
        for frag in fragments:
            mol = Chem.MolFromSmiles(frag)
            if mol is None:
                return False
        return True
    else:
        mol = Chem.MolFromSmiles(smi)
        return mol is not None

def canonicalize(smi, is_retro=False):
    """Canonicalize SMILES. For retro, canonicalize each fragment separately."""
    if is_retro:
        fragments = smi.split('.')
        canon_frags = []
        for frag in fragments:
            mol = Chem.MolFromSmiles(frag)
            if mol:
                canon_frags.append(Chem.MolToSmiles(mol))
            else:
                return ""
        return ".".join(sorted(canon_frags))  # Sort for consistent comparison
    else:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            return Chem.MolToSmiles(mol)
        return ""

def evaluate_sample(model_path, data_path, vocab_file, src_test_file, tgt_test_file, 
                    beam_size=10, n_best=10, temperature=2.75, batch_size=8, 
                    sample_size=50, max_length=512, min_length=1, is_retro=False):
    """Evaluate on a sample of the test set with proper validation."""
    
    print(f"[INFO] Loading model from {model_path}")
    with open(f"{model_path}/parameters.json", 'r') as f:
        config_dict = json.load(f)
    config = Box(config_dict)
    
    vocab = load_vocab(f'{data_path}/{vocab_file}')
    vocab_rev = [k for k, v in sorted(vocab.items(), key=lambda tup: tup[1])]
    
    rxng = RXNGraphormer("sequence_generation", config, vocab)
    model = rxng.get_model()
    
    ckpt_file = f"{model_path}/model/valid_checkpoint.pt"
    ckpt_inf = torch.load(ckpt_file, map_location=device, weights_only=False)
    model.load_state_dict(update_dict_key(ckpt_inf['model_state_dict']))
    model.to(device)
    model.eval()
    print("[INFO] Model loaded successfully!")
    
    # Load test dataset
    print(f"[INFO] Loading test dataset from {data_path}")
    test_dataset = RXNG2SDataset(root=data_path,
                                src_file=src_test_file,
                                tgt_file=tgt_test_file,
                                vocab_file=vocab_file,
                                trunck=0, multi_process=False, oh=False)
    
    # Take a sample
    total_samples = len(test_dataset)
    if sample_size < total_samples:
        indices = list(range(0, total_samples, total_samples // sample_size))[:sample_size]
        test_dataset = torch.utils.data.Subset(test_dataset, indices)
        print(f"[INFO] Using sample of {len(test_dataset)} examples (from {total_samples} total)")
    else:
        print(f"[INFO] Using full test set of {total_samples} examples")
    
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Get ground truth SMILES for the sample
    if hasattr(test_dataset, 'dataset'):  # Subset
        ground_truth_smiles_lst = []
        for idx in test_dataset.indices:
            tgt_token_ids = test_dataset.dataset[idx].tgt_token_ids[0]
            tgt_lens = test_dataset.dataset[idx].tgt_lens[0]
            smi = "".join([vocab_rev[idx] for idx in tgt_token_ids[:int(tgt_lens)-1]])
            ground_truth_smiles_lst.append(smi)
    else:
        ground_truth_smiles_lst = []
        for idx in range(len(test_dataset)):
            tgt_token_ids = test_dataset[idx].tgt_token_ids[0]
            tgt_lens = test_dataset[idx].tgt_lens[0]
            smi = "".join([vocab_rev[idx] for idx in tgt_token_ids[:int(tgt_lens)-1]])
            ground_truth_smiles_lst.append(smi)
    
    # Canonicalize ground truth
    ground_truth_canon = []
    for smi in ground_truth_smiles_lst:
        canon = canonicalize(smi, is_retro=is_retro)
        ground_truth_canon.append(canon)
    
    print(f"[INFO] Running inference with beam_size={beam_size}, n_best={n_best}, temperature={temperature}")
    all_predictions = []
    with torch.no_grad():
        for batch_data in test_dataloader:
            batch_data = batch_data.to(device)
            results = model.infer(reaction_batch=batch_data,
                                  batch_size=len(batch_data.tgt_lens),
                                  beam_size=beam_size,
                                  n_best=n_best,
                                  temperature=temperature,
                                  min_length=min_length,
                                  max_length=max_length)
            
            for predictions in results["predictions"]:
                smis = []
                for prediction in predictions:
                    predicted_idx = prediction.detach().cpu().numpy()
                    predicted_tokens = [vocab_rev[idx] for idx in predicted_idx[:-1]]
                    smi = "".join(predicted_tokens)
                    smis.append(smi)
                all_predictions.append(smis)
    
    # Compute metrics with proper validation
    n_samples = len(ground_truth_canon)
    
    # Standard accuracy (invalid predictions count as wrong)
    accuracies = np.zeros([n_samples, n_best], dtype=np.float32)
    
    # Valid prediction tracking
    valid_at_k = np.zeros([n_samples, n_best], dtype=bool)  # whether at least one valid in top-k
    first_valid_rank = np.full(n_samples, -1, dtype=int)     # rank of first valid prediction (0-indexed)
    
    for i in range(n_samples):
        smi_tgt = ground_truth_canon[i]
        if not smi_tgt:
            continue
        preds = all_predictions[i]
        
        found_match = False
        for j, smi in enumerate(preds):
            is_valid = validate_smiles(smi, is_retro=is_retro)
            if is_valid and first_valid_rank[i] == -1:
                first_valid_rank[i] = j
            
            if is_valid:
                valid_at_k[i, j:] = True
            
            if not found_match and is_valid:
                pred_canon = canonicalize(smi, is_retro=is_retro)
                if pred_canon == smi_tgt:
                    accuracies[i, j:] = 1.0
                    found_match = True
    
    # Compute metrics
    print("\n=== Standard Accuracy (invalids count as wrong) ===")
    for k in [1, 3, 5, 10, 20, 30]:
        if k <= n_best:
            acc = np.mean(accuracies[:, k-1])
            print(f"Top-{k} Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    
    # Valid prediction rate
    print("\n=== Valid Prediction Rate (at least 1 valid in top-k) ===")
    for k in [1, 3, 5, 10]:
        if k <= n_best:
            rate = np.mean(valid_at_k[:, k-1])
            print(f"Top-{k} Valid Rate: {rate:.4f} ({rate*100:.2f}%)")
    
    # Top-1 valid rate
    top1_valid = np.sum(first_valid_rank == 0)
    print(f"\nTop-1 Fully Valid Rate: {top1_valid}/{n_samples} = {top1_valid/n_samples:.4f} ({top1_valid/n_samples*100:.2f}%)")
    
    # Accuracy on valid predictions only (for samples where top-1 is valid)
    if top1_valid > 0:
        valid_samples_mask = (first_valid_rank == 0)
        acc_on_valid = np.mean(accuracies[valid_samples_mask, 0])
        print(f"Top-1 Accuracy (on valid top-1 only): {acc_on_valid:.4f} ({acc_on_valid*100:.2f}%)")
    
    # Detailed per-sample results for qualitative analysis
    results_detail = []
    for i in range(n_samples):
        gt = ground_truth_canon[i]
        preds = all_predictions[i]
        pred_details = []
        for j, smi in enumerate(preds):
            is_valid = validate_smiles(smi, is_retro=is_retro)
            canon = canonicalize(smi, is_retro=is_retro) if is_valid else ""
            match = (canon == gt) if is_valid else False
            pred_details.append({
                'rank': j+1,
                'smiles': smi,
                'valid': is_valid,
                'canonical': canon,
                'match': match
            })
        results_detail.append({
            'ground_truth': gt,
            'predictions': pred_details
        })
    
    return {
        'accuracies': accuracies,
        'predictions': all_predictions,
        'ground_truth': ground_truth_canon,
        'valid_at_k': valid_at_k,
        'first_valid_rank': first_valid_rank,
        'detail': results_detail
    }

if __name__ == "__main__":
    # Forward prediction (USPTO_STEREO)
    print("=" * 60)
    print("FORWARD PREDICTION EVALUATION (USPTO_STEREO)")
    print("=" * 60)
    
    model_path = "/run/csi/mount-root/nas/4079184d856ecc166ed19d4887083405/workspaces/default/rxngraphormer-web/models/seq-v2-USPTO_STEREO-20250509_070206_ft"
    data_path = "/run/csi/mount-root/nas/4079184d856ecc166ed19d4887083405/workspaces/default/rxngraphormer-web/rxngraphormer/dataset/USPTO_STEREO"
    
    forward_results = evaluate_sample(
        model_path=model_path,
        data_path=data_path,
        vocab_file="vocab_smiles.txt",
        src_test_file="src-test.txt",
        tgt_test_file="tgt-test.txt",
        beam_size=10,
        n_best=10,
        temperature=2.75,
        batch_size=8,
        sample_size=50,
        is_retro=False
    )
    
    with open("/run/csi/mount-root/nas/4079184d856ecc166ed19d4887083405/workspaces/default/rxngraphormer-web/eval_forward_results_fixed.pkl", "wb") as f:
        pickle.dump(forward_results, f)
    
    print("\n" + "=" * 60)
    print("RETROSYNTHESIS EVALUATION (USPTO_50k)")
    print("=" * 60)
    
    # Retrosynthesis (USPTO_50k)
    model_path_retro = "/run/csi/mount-root/nas/4079184d856ecc166ed19d4887083405/workspaces/default/rxngraphormer-web/models/USPTO_50k"
    data_path_retro = "/run/csi/mount-root/nas/4079184d856ecc166ed19d4887083405/workspaces/default/rxngraphormer-web/rxngraphormer/dataset/USPTO_50k"
    
    retro_results = evaluate_sample(
        model_path=model_path_retro,
        data_path=data_path_retro,
        vocab_file="vocab_smiles.txt",
        src_test_file="src-test.txt",
        tgt_test_file="tgt-test.txt",
        beam_size=10,
        n_best=10,
        temperature=3.0,
        batch_size=8,
        sample_size=50,
        is_retro=True
    )
    
    with open("/run/csi/mount-root/nas/4079184d856ecc166ed19d4887083405/workspaces/default/rxngraphormer-web/eval_retro_results_fixed.pkl", "wb") as f:
        pickle.dump(retro_results, f)