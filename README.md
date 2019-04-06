**Chobot project image classification model with API**  
Model for image classification custom trainening

**Push to local Docker registry**  
```
docker build -t chobot_image_custom_classification:latest . 
docker tag chobot_image_custom_classification:latest localhost:5000/chobot_image_custom_classification:latest
docker push localhost:5000/chobot_image_custom_classification:latest
```

**Push to Docker hub**  
```
docker build -t chobot_image_custom_classification:latest . 
docker tag chobot_image_custom_classification:latest stejsky/chobot_image_custom_classification:latest
docker push stejsky/chobot_image_custom_classification:latest
```

