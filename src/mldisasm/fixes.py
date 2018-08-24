#!/usr/bin/env python3

'''
Bugfixes and workarounds.
'''

def fix_output_size(config, tokens):
    '''
    Workaround for Disassembler's output_size parameter. The model has one output per token but the value in config
    (from config.json) doesn't track the actual number of tokens.
    '''
    config['model']['output_size'] = len(tokens)
