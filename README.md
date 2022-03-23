


```

materials used for dicussion with Spensor on his gan project - 2022-03-23


+ build container - 5min
```
bash build.sh
```

+ image download from tcia (contrast, no-contrast) 20 min.
```
export MYDATA=/radraid/pteng/tmp
docker run -it -v $PWD:/workdir -w /workdir -v $MYDATA:/mydata -p 6006:6006 cyclegan-ct-abdomen bash
python download.py sample.csv /mydata/raw
# api not functioning today https://github.com/TCIA-Community/TCIA-API-SDK/pull/1#issuecomment-1076581509
# instead used java app, `NBIA Data Retriever`
```

+ create csv file to seperate phases.

```
cd prepare
python prepare.py /mydata/c4kc-kits

```

+ train using published/available cyclegan. see `model-0`

```

```
