#!/usr/bin/env python3
"""
Evaluate RXNGraphormer on a sample of test data to compute top-k accuracy.
This is a lighter version that runs on a subset of the test set.
"""
import json
import torch
import numpy as np
from rdkit import Chem
from box import Box
import sys
sys.path.insert(0, '/run/csi/mount-root/nas/4079184d856ecc166ed19d4887083405/workspaces/default/rxngraphormer-web/rxngraphormer')

from rxngraphormer.model import RXNGraphormer
from rxngraphormer.data import load_vocab, RXNG2SDataset, smi_tokenizer
from rxngraphormer.utils import canonical_smiles, update_dict_key
from torch_geometric.loader import DataLoader

device = torch.device('cpu')

def evaluate_sample(model_path, data_path, vocab_file, src_test_file, tgt_test_file, 
                    beam_size=30, n_best=30, temperature=2.75, batch_size=16, 
                    sample_size=100, max_length=512, min_length=1):
    """Evaluate on a sample of the test set."""
    
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
        mol = Chem.MolFromSmiles(smi)
        if mol:
            ground_truth_canon.append(Chem.MolToSmiles(mol))
        else:
            ground_truth_canon.append("")
    
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
    
    # Compute accuracies
    accuracies = np.zeros([len(ground_truth_canon), n_best], dtype=np.float32)
    for i in range(len(ground_truth_canon)):
        smi_tgt = ground_truth_canon[i]
        if not smi_tgt:
            continue
        preds = all_predictions[i]
        preds_canon = []
        for smi in preds:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                preds_canon.append(Chem.MolToSmiles(mol))
            else:
                preds_canon.append("")
        for j, smi in enumerate(preds_canon):
            if smi == smi_tgt:
                accuracies[i, j:] = 1.0
                break
    
    # Print results
    print("\n=== Accuracy Results ===")
    for k in [1, 3, 5, 10, 20, 30]:
        if k <= n_best:
            acc = np.mean(accuracies[:, k-1])
            print(f"Top-{k} Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    
    return accuracies, all_predictions, ground_truth_canon

if __name__ == "__main__":
    # Forward prediction (USPTO_STEREO)
    print("=" * 60)
    print("FORWARD PREDICTION EVALUATION (USPTO_STEREO)")
    print("=" * 60)
    
    model_path = "/run/csi/mount-root/nas/4079184d856ecc166ed19d4887083405/workspaces/default/rxngraphormer-web/models/seq-v2-USPTO_STEREO-20250509_070206_ft"
    data_path = "/run/csi/mount-root/nas/4079184d856ecc166ed19d4887083405/workspaces/default/rxngraphormer-web/rxngraphormer/dataset/USPTO_STEREO"
    
    acc_forward, preds_forward, gt_forward = evaluate_sample(
        model_path=model_path,
        data_path=data_path,
        vocab_file="vocab_smiles.txt",
        src_test_file="src-test.txt",
        tgt_test_file="tgt-test.txt",
        beam_size=10,
        n_best=10,
        temperature=2.75,
        batch_size=8,
        sample_size=50,  # Smaller sample for faster evaluation
    )
    
    # Save results
    import pickle
    with open("/run/csi/mount-root/nas/4079184d856ecc166ed19d4887083405/workspaces/default/rxngraphormer-web/eval_forward_results.pkl", "wb") as f:
        pickle.dump({
            'accuracies': acc_forward,
            'predictions': preds_forward,
            'ground_truth': gt_forward
        }, f)
    
    print("\n" + "=" * 60)
    print("RETROSYNTHESIS EVALUATION (USPTO_50k)")
    print("=" * 60)
    
    # Retrosynthesis (USPTO_50k)
    model_path_retro = "/run/csi/mount-root/nas/4079184d856ecc166ed19d4887083405/workspaces/default/rxngraphormer-web/models/USPTO_50k"
    data_path_retro = "/run/csi/mount-root/nas/4079184d856ecc166ed19d4887083405/workspaces/default/rxngraphormer-web/rxngraphormer/dataset/USPTO_50k"
    
    acc_retro, preds_retro, gt_retro = evaluate_sample(
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
    )
    
    with open("/run/csi/mount-root/nas/4079184d856ecc166ed19d4887083405/workspaces/default/rxngraphormer-web/eval_retro_results.pkl", "wb") as f:
        pickle.dump({
            'accuracies': acc_retro,
            'predictions': preds_retro,
            'ground_truth': gt_retro
        }, f)