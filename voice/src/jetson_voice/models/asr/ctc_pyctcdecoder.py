#!/usr/bin/env python3
# coding: utf-8

import os
import logging

from .ctc_decoder import CTCDecoder
from .ctc_utils import find_silent_intervals, merge_words, rebase_word_times, split_words, transcript_from_words

from jetson_voice.utils import global_config

from pyctcdecode import build_ctcdecoder 

class CTCPyCTCDecoder(CTCDecoder):
    """
    CTC beam search decoder that optionally uses a language model.
    """
    def __init__(self, config, vocab, resource_path=None):
        """
        Create a new CTCBeamSearchDecoder.
        
        See CTCDecoder.from_config() to automatically create
        the correct type of instance dependening on config.
        """
        super().__init__(config, vocab)
        self.config.setdefault('word_threshold', -1000.0)
        self.reset()
        
        # set default config
        # https://github.com/NVIDIA/NeMo/blob/855ce265b80c0dc40f4f06ece76d2c9d6ca1be8d/nemo/collections/asr/modules/beam_search_decoder.py#L21
        self.config.setdefault('language_model', None)
        self.config.setdefault('beam_width', 128)#)
        self.config.setdefault('alpha', 0.8 if self.language_model else 0.0)
        self.config.setdefault('beta', 0.0)
        self.config.setdefault('cutoff_prob', 1.0)
        self.config.setdefault('cutoff_top_n', 40)
        self.config.setdefault('top_k', 3)
        
        # check for language model file
        if self.language_model:
            if not os.path.isfile(self.language_model):
                self.config['language_model'] = os.path.join(resource_path, self.language_model)
                if not os.path.isfile(self.language_model):
                    raise IOError(f"language model file '{self.language_model}' does not exist")
                    
        logging.info('creating CTCBeamSearchDecoder')
        logging.info(str(self.config))
        
        self._decoder = build_ctcdecoder(
            labels=self.vocab,
            alpha=self.config['alpha'],
            beta=self.config['beta'],
            kenlm_model_path=self.language_model
        )

        self.hotword_weight = 1
        # TODO: mover isso para um arquivo de configuração
        self.hotwords = ["zordon", "follow", "me", "often", "creator", "favorite", 
        "is", "in", "hitchbot", "ever", "are", "allowed", "it's", 
        "style", "what", "for", "movies", "away", "mark", "ar", "di" 
        "python", "call", "music", "you", "has", "was", "do", "safe", "run", 
        "get", "self", "driving", "cars", "why", "who", "robots", "zuckerberg", 
        "language", "lunch", "be", "invented", "killed", "compiler", "kind", "ate", 
        "your", "like", "salad", "created", "did", "of", "programming", "robot", 
        "shouldn't", "person", "so", "the", "angry", "yes", "no", "up", "down", "left", "we", 
        "on", "off", "to", "go"]
            
    def decode(self, logits, **hotwords_kw):
        """
        Decode logits into words, and merge the new words with the
        previous words from the running transcript.
        """
        hotwords_kw.setdefault("hotwords", self.hotwords)
        hotwords_kw.setdefault("hotword_weight", self.hotword_weight)
        
        beams = self._decoder.decode_beams(logits, **hotwords_kw)
        if global_config.debug: print("beams", beams)

        score = beams[0][-1]
        
        text = beams[0][0]
        if global_config.debug: print("text", text)
        
        chunk_offset = beams[0][2]
        word_offsets = []
        for word, (start_offset, end_offset) in chunk_offset:
            word_offsets.append({"word": word, "start_offset": start_offset, "end_offset": end_offset})

        words = [{
            'text' : item["word"],
            'start_time' : item["start_offset"],
            'end_time' : item["end_offset"],
            'score': score
        } for item in word_offsets]
        if global_config.debug: print("words", words)

        # merge new words with past words
        self.words = merge_words(self.words, words, self.config['word_threshold'], 'similarity')
        if global_config.debug: print("self.words", self.words)

        # look for silent/EOS intervals
        silent_intervals = find_silent_intervals(logits, len(self.vocab), self.timesteps_silence, self.timestep) 
        
        if global_config.debug: 
            print(f'silent intervals:  {silent_intervals}')

        self.timestep += self.timestep_delta
        
        # split the words at EOS intervals
        if len(silent_intervals) > 0:
            wordlists = split_words(self.words, silent_intervals)
            transcripts = []
            
            for idx, wordlist in enumerate(wordlists):
                # ignore blanks (silence after EOS has already occurred)
                if len(wordlist) == 0:
                    continue
                    
                # if there is only one wordlist, then it must be EOS
                # if there are multiple, then the last one is not EOS
                end = (len(wordlists) == 1) or (idx < (len(wordlists) - 1))
                
                if end:
                    wordlist = rebase_word_times(wordlist)
                    self.reset()            # TODO reset timesteps counter correctly
                else:
                    self.words = wordlist   
                    
                transcripts.append((wordlist, end))
        else:
            transcripts = [(self.words, False)]

        return [{
            'text' : transcript_from_words(words, scores=global_config.debug, times=global_config.debug, end=end, add_punctuation=self.config['add_punctuation']),
            'words' : words,
            'end' : end
        } for words, end in transcripts]
        
    def reset(self):
        """
        Reset the CTC decoder state at EOS (end of sentence)
        """
        #self.timestep = 0
        #self.tail_silence = 0
        self.words = []
        
    @property
    def language_model(self):
        return self.config['language_model']
 