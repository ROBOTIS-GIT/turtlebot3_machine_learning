echo "COMPOSE UP"
sudo docker-compose up --scale ros_ml=2 &
#echo SLEEP
sleep 3

echo "SET VARIABLE"
CONTAINERID=$(sudo docker ps | grep turtlebot3_machine_learning_docker_ros_ml| awk '{print $1}') 

echo $CONTAINERID
#echo "CHECK VALUE"
#echo "ID:" > test.txt 
#echo $CONTAINERID >> test.txt
#cat test.txt

#echo "EXEC"
#sudo docker exec -it $CONTAINERID /bin/bash

