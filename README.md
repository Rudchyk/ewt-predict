- [Coindesk (Energy Web Token)](https://www.coindesk.com/price/ewt)
- [ewtPredict.ipynb](https://colab.research.google.com/drive/1d7-866a2WzmML39Et0j84N2G5ued9vYY?usp=sharing)

## Setup

```
py --version
which python
pip install -r requirements.txt

```

## Convert LLM

```
py -m tf2onnx.convert --saved-model ./llm/model_1_dense --output ./llm/model_1_dense.onnx --opset 13
```
