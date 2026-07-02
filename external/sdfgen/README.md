# Local SDFGen Binaries

Put the SDFGen executable used by preprocessing here:

```text
external/sdfgen/
  computeDistanceField
```

The preprocessing script automatically uses this path when it exists. If the binary depends on bundled shared libraries, keep them beside it, for example:

```text
external/sdfgen/
  computeDistanceField
  tbb/
  libtcmalloc.so.4
  libglut.so.3
```

The binaries and shared libraries in this directory are committed with the project for reproducible preprocessing. They are Linux x86-64 binaries from the SDFusion preprocessing `isosurface` folder, so they are expected to run on Linux servers such as AutoDL rather than macOS.

Source:

```text
https://github.com/yccyenchicheng/SDFusion/tree/master/preprocess/isosurface
commit 09801f6d04c04816e5827af981e2ae77be77ce5b
```

The copied SDFusion license is stored in `LICENSE.SDFusion`.

Verify on Linux:

```bash
chmod +x external/sdfgen/computeDistanceField
ldd external/sdfgen/computeDistanceField
external/sdfgen/computeDistanceField
```

If `ldd` or preprocessing reports `libGLU.so.1: cannot open shared object file`, install the OpenGL utility library on the server:

```bash
apt-get update
apt-get install -y libglu1-mesa
```

If system packages are unavailable, install into the active conda environment instead:

```bash
conda install -c conda-forge libglu freeglut -y
```

Use explicitly:

```bash
python tools/preprocess_shapenet_obj_to_sdf.py \
  --shapenet_root /root/autodl-tmp/data/ShapeNetCore.v1 \
  --data_root /root/autodl-tmp/data \
  --category chair \
  --backend sdfgen \
  --sdfgen external/sdfgen/computeDistanceField \
  --write_filelist
```
