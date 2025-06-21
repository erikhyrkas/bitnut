
# BitNut

![bitnut.png](bitnut.png)

A fun little experiment to make a bitnet-based llm that talks like a dog who believes the world is run by squirrels.

Currently, the model is about 133M parameters.

Credit: There's not a ton of unique thought that went into this. I thought this paper from Microsoft (https://arxiv.org/html/2504.12285v2) was nifty and wanted to see what I could do locally. 

This project is strictly about curiosity and not really useful for any real-life purpose beyond amusement and learning.

# corpus

In pre-training, used the cosmopedia-v2 from 
https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus 

For finetuning, we generated our own data using the original bitnet 1.58, because... it's fast and a bit of a nod to it.
We could have used a bigger or more powerful model, but it would take longer and this is just for fun. Also, we know that 
finetuning isn't as good as reinforcement learning, but finetuning is simple to do and doesn't require as much experimenting
to get right. I see a number of examples as I scan through where the original bitnet didn't follow instructions perfectly--
like confusing talking to the dog named BitNut vs being the dog BitNut. I'm also sure that it doesn't always have its facts
straight. That said, this is for fun.

The original bitnet 1.58 had three training phases: large-scale pre-training followed by supervised fine-tuning (SFT) and direct preference optimization (DPO). 
It used:
* pretraining: DCLM, FineWeb-EDU, and they hinted at other sources in the paper -- like generating math samples as well.
* sft: WildChat, LMSYS-Chat-1M, WizardLM Evol-Instruct, SlimOrca, plus generated datasets using GLAN and MathScale 
* DPO: UltraFeedback and MagPie

The original also did pretraining in two parts. Part one had a higher learning rate and threw most of the web at the model. The second "cooldown" pretraining was higher quality content at a smaller learning rate.


# architecture

Just took the design from here and stripped it down to an even smaller model: https://huggingface.co/microsoft/bitnet-b1.58-2B-4T

# Chat template

I did not get fancy here. I could have put more effort into making something robust, but frankly, this is for fun 
and we don't need anything fancy. Being compact and simple is useful in such a small model anyway.

Template: `human: {user_input}\nbitnut:`

# setup

PyTorch with cuda support:

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

Bitnet's transformers (https://github.com/shumingma/transformers):

pip3 install git+https://github.com/shumingma/transformers.git

additional installs:

pip3 install hf_xet

pip3 install -r requirements.txt


# steps

1. train_tokenizer.py (let's make our own tokenizer...because we can.)
2. pretrain.py (Give us a base model. This will download a huge amount of data. I hope you have disk.)
3. run-pretrain.py (should be able to see the base model works, even if it's not great.)
4. generate_prompts.py (Before we can fine tune, we need a bunch of example prompts.) 
5. apply_bitnut_voice.py (make finetuning training data, with responses to the prompts in bitnut's voice.)
6. finetune.py (Make a finetuned model with bitnut's voice.)
7. run.py (chat with bitnut)

# Notes
* Loss plateaued around 3.2 billion tokens (0.64 through first epoch) (~2406 tokens per parameter)
  * the chinchilla ratio (20 tokens per parameter is not terribly relevant to small models)
  * other small models have needed about `10k tokens per parameter`
  * I feel like it has trained well to this point: when I run the model at this check point, it has basic english fluency (more or less) but it doesn't really have great understanding or focus. It makes mostly syntactically correct sentences, but meanders between sentences. It seems vaguely aware of the starting topic, but could veer off wildly to new topics. It makes up facts with wild abandon.  
  * I probably should have used a cosine scheduler
  * I probably should have used 2 epochs with 10 billion tokens in each rather than 3 with 5 billion in each
  * Maybe even 1 epoch of 10 billion tokens from cosmopedia-v2 and then 1 epoch of 10 billion tokens from fineweb-edu
