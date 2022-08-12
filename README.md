# HPX Project

description tbd.

## Docker Containers

### Release (HPX installed in Release mode)

Build image inside hpx_project folder:

`sudo docker build . -f docker/release/Dockerfile -t hpx_release_image`

Start container:

`sudo docker run -it --rm --user user hpx_release_image`


### Debug (HPX installed in Debug mode)

Build image inside hpx_project folder:

`sudo docker build . -f docker/debug/Dockerfile -t hpx_debug_image`

Run container inside hpx_project folder:

`sudo docker run -it --rm --user user  hpx_debug_image`


### Base (Install HPX manually)

Build image inside hpx_project folder:

`sudo docker build . -f docker/base/Dockerfile -t hpx_base_image`

Run container inside hpx_project folder:

`sudo docker run -it --rm --user user hpx_base_image`

Install HPX manually:

`cd && hpx_project/dependencies && git pull --rebase && ./build-all.sh Release`


## Compile Code Manually

### Release

Compile Code:

`cd && cd hpx_project && rm -rf build && git pull --rebase && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make all`

### Debug

Compile Code:

`cd && cd hpx_project && rm -rf build && git pull --rebase && mkdir build && cd build && cmake .. && make all`

## Run Code Manually

Run Code:

`cd && cd hpx_project/build && ./hpx_cholesky --n_train 1000 --n_test 1000 --n_regressors 100 --n_tiles 1 --cholesky left`


## Scripts

Tile Number Loop:

`cd && cd hpx_project/benchmark_scripts && ./tiles_script.sh 10 100 10 1000 1000 100 left 5`

Training Size Loop:

`cd && cd hpx_project/benchmark_scripts && ./data_script.sh 1000 5000 1000 10 1000 100 left 5`

Benchmark Script:

`cd && cd hpx_project && git pull --rebase && ./run_benchmarks.sh`


## Git Commands

`git add . && git commit -m "comment" && git push`

`git add . && git commit --amend --no-edit && git push -f`
