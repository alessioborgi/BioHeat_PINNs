# BIOHEAT_PINNs: Temperature Estimation with Bio-Heat Equation using Physics-Informed Neural Networks

The project focuses on the real-time estimation of temperature distribution in biological tissues using the Pennes Bio-Heat Equation, a fundamental model for heat transfer in human tissues. This research applies Physics-Informed Neural Networks (PINNs), a deep learning framework designed to solve partial differential equations (PDEs) by embedding physical laws directly into the learning process. The primary objective of the project is to enhance the control and predictability of hyperthermia treatments—a form of thermal therapy used to treat conditions such as tumors by precisely heating affected tissues.

Hyperthermia therapy requires maintaining tissue temperature within a specific range (typically between 39°C and 45°C) for prolonged periods. However, this process is challenging due to the dynamic heat transfer between tissues and the circulatory system, as well as the risk of damaging healthy tissues. This project simplifies the resolution of the Pennes Bio-Heat equation in 1D and 2D spatial domains using PINNs to provide an efficient and accurate tool for predicting temperature distribution in both cutaneous (surface-level) and subcutaneous (beneath the skin) tissues.


## CONTAINER INSTRUCTIONS 


#### These are the commands to run in the Terminal to build the Container and so on.
- Open **Docker** 

- Enter the folder ProjectMedicalRobotics:
    ```cd ProjectMedicalRobotics```

- Build the Container, with arm64 for MAC Ms Processor:

    ```docker build --platform linux/arm64 -t bio_heat_pinns .```
    
- To run the Container without re-building it every time. It saves the changes instantaneously.

    ```docker run --platform linux/arm64 -it --name bio_heat_pinns -v "$(pwd):/working_dir" bio_heat_pinns```

    With this command you are inside the Container and you are able to Run the files.
    
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
Once you are inside the Container you can run the files inside the repository:

- Before the First run you will have to select one of the three option regarding the **WandB** account and upload your API Key.

- To Run the main.py file:
    ```python3 ./src/main.py```

- To Run the tuning.py file:
    ```python3 ./src/tuning.py```

