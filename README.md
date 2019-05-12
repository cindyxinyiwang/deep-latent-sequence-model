# TextStyleTransfer

## Data

* `data/yelp` is the preprocessed Yelp dataset which meets our format requirement. `data/yelp_shen` is the original dataset used in https://arxiv.org/abs/1705.09655

Run `prepare_data.py` to prepare data:

```shell
python scripts/prepare_data.py --dataset [dataset_name]
```
where `--dataset` currently supports choices `{yelp,formality,decipher}`.

## Requirements
* Python 3, PyTorch 0.4
* [tqdm](https://github.com/tqdm/tqdm)

## LM Usage
Train:
```shell
CUDA_VISIBLE_DEVICES=xx python src/lm_lstm.py --dataset [dataset name] --style [binary style indicator]
```

Eval: see `scripts/decipher/sanity_check_lm_decipher.sh`

## TODO

* ~~(junxian): noise module with and without bpe encoding~~
* ~~(cindy): bpe preprocessed dataset ?~~
* ~~(cindy): add sampling decoding options in back-translation~~
* (junxian): restrict each batch has the same y would accelarate LM prior computation


## Pre-trained LM test PPL
* Yelp:

  | train\test | style 0 | style 1 |
  | ----------- | ------- | ------- |
  | **model style 0** | 31.97   | 70.04   |
  | **model style 1** | 119.19  | 21.87   |

* Decipherment 100%

  | train\test | style 0 | style 1 |
  | ----------- | ------- | ------- |
  | **model style 0** | 30.48  | 956140.07  |
  | **model style 1** | 672110.37  | 30.35   |

* sr_bos bpe vocab32000

  | train\test | style 0 | style 1 |
  | ----------- | ------- | ------- |
  | **model style 0** | 94.46  | 10438.70  |
  | **model style 1** |  159658.92 |  131.20  |

* form_em

  | train\test | style 0 | style 1 |
  | ----------- | ------- | ------- |
  | **model style 0** | 71.30  | 356.31  |
  | **model style 1** |  170.26 | 135.50|

* shakespeare

  | train\test | style 0 | style 1 |
  | ----------- | ------- | ------- |
  | **model style 0** | 132.95  | 175.21  |
  | **model style 1** |  363.10 | 85.25|
  

## References
* [What is wrong with style transfer for texts?](https://arxiv.org/abs/1808.04365)
* [Dear Sir or Madam, May I introduce the GYAFC Dataset: Corpus, Benchmarks and Metrics for Formality Style Transfer](https://arxiv.org/abs/1803.06535)
* [Evaluating Style Transfer for Text](https://arxiv.org/pdf/1904.02295)
* [Unsupervised Text Style Transfer via Iterative Matching and Translation](https://arxiv.org/pdf/1901.11333)
