## Build 

```bash
docker run --rm -v $PWD:/project -w /project --device=/dev/ttyUSB0 --privileged -it espressif/idf:v5.3 idf.py -p /dev/ttyUSB0 build
```

## Flash

```bash
docker run --rm -v $PWD:/project -w /project --device=/dev/ttyUSB0 --privileged -it espressif/idf:v5.3 idf.py -p /dev/ttyUSB0 flash
```

## Monitor

```bash
docker run --rm -v $PWD:/project -w /project --device=/dev/ttyUSB0 --privileged -it espressif/idf:v5.3 idf.py -p /dev/ttyUSB0 monitor
```
