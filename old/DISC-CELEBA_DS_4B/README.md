To run:
- Create folder: ./save
- Run ```THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python control.py```

To run ANIME:
- Run ```THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python celeba_mn.py -S /data/lisa/data/anime_faces.hdf5 -o ./output/ -n anime```