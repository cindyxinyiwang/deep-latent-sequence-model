# TextStyleTransfer

## Data

* `data/yelp` is the preprocessed Yelp dataset which meets our format requirement. `data/yelp_shen` is the original dataset used in https://arxiv.org/abs/1705.09655

Run `prepare_data.py` to prepare data:

```shell
python scripts/prepare_data.py --dataset [dataset_name]
```
where `--dataset` currently supports choices `{decipher}`.

## Requirements
* Python 3, PyTorch 0.4
* [tqdm](https://github.com/tqdm/tqdm)

## TODO

* ~~(junxian): noise module with and without bpe encoding~~
* ~~(cindy): bpe preprocessed dataset ?~~
* ~~(cindy): add sampling decoding options in back-translation~~
* (junxian): restrict each batch has the same y would accelarate LM prior computation


## Pre-trained LM test PPL
* Yelp:

  | train\stest | style 0 | style 1 |
  | ----------- | ------- | ------- |
  | **style 0** | 31.97   | 70.04   |
  | **style 1** | 119.19  | 21.87   |

  


## References
* [What is wrong with style transfer for texts?](https://arxiv.org/abs/1808.04365)
* [Dear Sir or Madam, May I introduce the GYAFC Dataset: Corpus, Benchmarks and Metrics for Formality Style Transfer](https://arxiv.org/abs/1803.06535)
