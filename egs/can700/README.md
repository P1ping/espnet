tts1-vanilla:
	can-data
	vanilla Tacotron
tts2-mixing:
	can-lj-data
	Tacotron + speaker embeddings (256-dim)
tts3-charembed:
	can-data
	Tacotron + character embeddings (768-dim)
	(To be trained)
tts4-intoembed:
	can-data
	Tacotron + intonation embeddings (256-dim)
	(To be trained)
tts5-intolearn:
	can-data
	Tacotron + character embeddings + type loss
	(To be designed)
	(To be trained)


cantts-data:
	Code-mixing Cantonese data
        CANTTS CMVN
	None
cantts-data-nopad:
	Code-mixing Cantonese data without padding
	CANTTS CMVN
	None
can-data:
	Pure Cantonese data
	CANTTS CMVN
	None
can-data-nopad:
	Pure Cantonese data without padding
	CANTTS CMVN
	None
cantts-lj-data:
        Code-mixing Cantonese data + LJSpeech data(0.2 number)
        CANTTS CMVN
        None
        (Data also used for vocoder training)
