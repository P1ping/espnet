#!/usr/bin/env python3

import sys

sys.path.append('/data1/baibing/tools/g2p_can')

import argparse
from g2p_can import G2P

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("text_file", type=str, help="Input file of text.")
    parser.add_argument("token_file", type=str, help="Output file of tokens.")
    parser.add_argument("--add_id", type=str, default="False", help="Whether add utterance ids to the head.")
    parser.add_argument("--sos", type=str, default=None, help="Starting token to add to the head.")
    parser.add_argument("--eos", type=str, default=None, help="Ending token to add to the head.")

    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    text_file = args.text_file
    token_file = args.token_file
    add_id = True if args.add_id == 'True' else False
    sos = args.sos
    eos = args.eos

    g2p = G2P()
    g2p.load_files()
    g2p.add_extra_char_map([' '], ['<space>'])

    token_lines = []
    utt_id = 1
    with open(text_file, 'r', encoding='utf8') as f:
        for line in f:
            text = line.strip()
            tokens = g2p.jyutping(text, unk_sym='<unk>')
            if not tokens:
                print("Invalid sentence (%s) is skipped" % text)
                continue
            if sos:
                tokens = [sos] + tokens
            if eos:
                tokens = tokens + [eos]
            if add_id:
                tokens = ["utt_%d" % utt_id] + tokens
            token_text = ' '.join(tokens) + '\n'
            token_lines += [token_text]
            utt_id += 1
    
    with open(token_file, 'w', encoding='utf8') as f:
        f.writelines(token_lines)
