# BIO-HEAT PINNs
## CONTAINER INSTRUCTIONS 

#### These are the commands to run in the Terminal to build the Container and so on.
- Open Docker 

- Enter the folder ProjectMedicalRobotics:
    ```cd ProjectMedicalRobotics```

- Build the Container, with arm64 for MAC Ms Processor:
    ```docker build --platform linux/arm64 -t bio_heat_pinns .```
    
- To run the Container without re-building it every time. It saves the changes instantaneously.
    ```docker run --platform linux/arm64 -it --name bio_heat_pinns -v "$(pwd):/working_dir" bio_heat_pinns```
    
- To Re-Start the Container (when closing and opening the pc, for example):
    ```docker run -it --name bio_heat_pinns```


## UTILS
- Stop the Container (from an external terminal): 
    ```docker stop bio_heat_pinns```

- Stop the Container (from the internal terminal):
    **ctrl + D**

- Remove the Container (from an external terminal): 
    ```docker rm bio_heat_pinns```


## FILE INSTRUCTIONS
- Before the First run you will have to select one of the three option regarding the **WandB** account and upload your API Key.

- To Run the main.py file:

    ```python3 ./src/main.py```

- To Run the tuning.py file:

    ```python3 ./src/tuning.py```

