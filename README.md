### 1. Install the package with Docker

##### Pre-requisites
- install [**docker**](https://docs.docker.com/get-docker/) and [**docker-compose**](https://docs.docker.com/compose/install/)
- clone the repo via `git clone -b git clone -b bisc_2024 https://github.com/sinzlab/nnvision.git`


### **Start Jupyterlab environment**
- create a `.env` file, on the basis of the `.env_example` 
- now you can create the docker container:
```
cd nnvision/
docker-compose run -d -p 10101:8888 notebook_server
```
to access the container, type in `localhost:10101` in any browser.


Example Notebook for recreating the plots can be found under: [**Example_plots**](notebooks/bisc_2024/plot_examples.ipynb)
