# HPX Project

Repository that contains the HPX implementation of the non-linear system
identification with Gaussian processes minimum working example used for
the research project:
"A comparison of PETSc and HPX for a Scientific Computation Problem"

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


## Compile and Run Code

Benchmark Script:

`cd && cd hpx_project && git pull --rebase && ./run_benchmarks.sh cpu/gpu/blas`


## Git Commands for Developing

`git add . && git commit -m "comment" && git push`

`git add . && git commit --amend --no-edit && git push -f`
