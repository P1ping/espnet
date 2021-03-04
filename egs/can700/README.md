tts1-vanilla:
	Pure Cantonese data
        LJ_MBMG CMVN
	vanilla Tacotron
	(Data also used for vocoder training)
tts1-vanilla-canttscmvn
	Pure Cantonese data
	CANTTS CMVN
	vanilla Tacotron
tts2-mixing:
	Code-mixing Cantonese data + LJSpeech data (half number)
	LJ_MBMG CMVN
	Tacotron + speaker embeddings (256-dim)
tts3-charembed:
	Pure Cantonese data
	X
	Tacotron + character embeddings (768-dim)
	(To be trained)
tts4-intoembed:
	Pure Cantonese data
	X
	Tacotron + intonation embeddings (256-dim)
	(To be trained)
tts5-intolearn:
	Pure Cantonese data
	X
	Tacotron + character embeddings + type loss
	(To be designed)
	(To be trained)


cantts-data:
	Code-mixing Cantonese data
        CANTTS CMVN
	None
cantts-lj-data:
	Code-mixing Cantonese data + LJSpeech data(0.2 number)
	CANTTS CMVN
	None
	(Data also used for vocoder training)
tts1-vanilla-canttscmvn
        Pure Cantonese data
        CANTTS CMVN
        None
