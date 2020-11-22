<h1>Steps to Run: </h1>

1. Unzip the resnet50_bceloss_final_model.zip:

        unzip resnet50_bceloss_final_model.zip
2. Create folder checkpoints:

        mkdir checkpoints
3. Move extracted model to checkpoints:

        mv resnet50_bceloss_final_model.pth ./checkpoints/
4. Run resnet50.py file as follows:

   For training:
   
        python resnet50.py --flag train
   For testing:
   
        python resnet50.py --flag test
