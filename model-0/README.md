

+ train

```
cd model-0
docker tag cyclegan-ct-abdomen $REMOTE_URL
docker push $REMOTE_URL
# create submit.condor and relevant sh file
mkdir joblog
condor_submit submit.condor
```


+ monitor

```

tensorboard --bind_all --logdir=log
```