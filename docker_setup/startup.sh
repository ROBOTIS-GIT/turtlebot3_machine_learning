echo "COMPOSE UP"
sudo docker-compose up --scale ros_ml=2 &
sleep 3

echo "SET VARIABLE"
CONTAINERID=$(sudo docker ps | grep docker_setup_ros_ml| awk '{print $1}')

echo $CONTAINERID
