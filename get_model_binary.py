import argparse
from train import KoBARTConditionalGeneration
from transformers.models.bart import BartForConditionalGeneration
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--hparams", default=None, type=str)
parser.add_argument("--model_binary", default=None, type=str)
parser.add_argument("--output_dir", default='kobart_summary', type=str)
args = parser.parse_args()

inf = KoBARTConditionalGeneration.load_from_checkpoint(args.model_binary)

inf.model.save_pretrained(args.output_dir)

