#MASTER
	MASTER=`spark-ec2 -k luke -i ~/luke.pem --region=eu-west-1 get-master NAME | tail -1`


#download all data onto workers
(ssh -i ~/luke.pem root@$MASTER 'cat /root/spark-ec2/slaves' && echo $MASTER) | while read line; do
	ssh -n -o StrictHostKeyChecking=no -i ~/luke.pem -t -t root@$line 'source aws_credentials.sh && /root/.local/lib/aws/bin/aws s3 cp s3://neuronforest.sparkdata /mnt/data --recursive' &
done


#run job
~/spark/bin/spark-submit --driver-memory 12G --conf spark.shuffle.spill=false --conf spark.shuffle.memoryFraction=0.4 --conf spark.storage.memoryFraction=0.4 --master spark://`cat ~/spark-ec2/masters`:7077 --class Main ./neuronforest-spark.jar data_root=/mnt/data master= subvolumes=.*36 dimOffsets=0 malisGrad=50 nTrees=10 iterations=2


#watch log for one worker
(ssh -i ~/luke.pem root@$MASTER 'tail -1 /root/spark-ec2/slaves') | while read line; do ssh -n -o StrictHostKeyChecking=no -i ~/luke.pem -t -t root@$line 'cd /root/spark/work/ && echo $(ls | tail -1) && cd $(ls | tail -1) && tail -f ./*/stdout' ; done



(ssh -i ~/luke.pem root@$MASTER 'cat /root/spark-ec2/slaves' && echo $MASTER) | while read line; do
	scp -o StrictHostKeyChecking=no -r -i ~/luke.pem root@$line:/masters_predictions/2014-12-29\ 02-29-39
/masters_predictions/ &
done


#clear existing predictions
(ssh -i ~/luke.pem root@$MASTER 'cat /root/spark-ec2/slaves' && echo $MASTER) | while read line; do
	ssh -n -o StrictHostKeyChecking=no -i ~/luke.pem -t -t root@$line 'rm -r /masters_predictions; mkdir /masters_predictions'
done

#download different data to each worker
#CLUSTER36
volumes=(im1/split_111/000
im2/split_111/000
im3/split_111/000
im4/split_111/000
im5/split_122/000
im5/split_122/001
im5/split_122/010
im5/split_122/011
im6/split_122/000
im6/split_122/001
im6/split_122/010
im6/split_122/011
im7/split_122/000
im7/split_122/001
im7/split_122/010
im7/split_122/011
im8/split_122/000
im8/split_122/001
im8/split_122/010
im8/split_122/011
im9/split_122/000
im9/split_122/001
im9/split_122/010
im9/split_122/011
im10/split_122/000
im10/split_122/001
im10/split_122/010
im10/split_122/011
im11/split_122/000
im11/split_122/001
im11/split_122/010
im11/split_122/011
im12/split_122/000
im12/split_122/001
im12/split_122/010
im12/split_122/011)
(echo $MASTER && ssh -n -i ~/luke.pem root@$MASTER 'cat /root/spark-ec2/slaves') | (tasks=""; for v in ${volumes[0]} ${volumes[@]}; do
	read line; ssh -n -o StrictHostKeyChecking=no -i ~/luke.pem -t -t root@$line "source aws-credentials && /usr/local/aws/bin/aws s3 cp s3://neuronforest.sparkdata/$v /mnt/data --recursive" &
tasks="$tasks $!"; done; for t in $tasks; do wait $t; done)

#copy predictions to s3
(ssh -n -i ~/luke.pem root@$MASTER 'cat /root/spark-ec2/slaves') | (while read line; do
	ssh -n -o StrictHostKeyChecking=no -i ~/luke.pem -t -t root@$line "source aws-credentials && /usr/local/aws/bin/aws s3 cp /masters_predictions/ s3://neuronforest.sparkdata/predictions --recursive" &
done)

#...FROM MASTER
(cat ~/spark-ec2/slaves) | (while read line; do ssh -n -o StrictHostKeyChecking=no -t -t root@$line "source aws-credentials && /usr/local/aws/bin/aws s3 cp /masters_predictions/ s3://neuronforest.sparkdata/predictions --recursive" & done)

#force stop
yes | spark-ec2 -i ~/luke.em --region eu-west-1 stop NAME

#s3 to home
aws s3 sync s3://neuronforest.sparkdata/predictions /masters_predictions/s3



#NAME
volumes=(
im1/split_112/000
im1/split_112/001
im2/split_112/000
im2/split_112/001
im3/split_112/000
im3/split_112/001
im4/split_111/000
)

#CLUSTER18
volumes=(
im1/split_111/000
im2/split_111/000
im5/split_112/000
im5/split_112/001
im6/split_112/000
im6/split_112/001
im7/split_112/000
im7/split_112/001
im8/split_112/000
im8/split_112/001
im9/split_112/000
im9/split_112/001
im10/split_112/000
im10/split_112/001
im11/split_112/000
im11/split_112/001
im12/split_112/000
im12/split_112/001
)





#RUN, COPY, DELETE:
set +H

~/spark/bin/spark-submit --driver-memory 12G --conf spark.shuffle.spill=false --conf spark.shuffle.memoryFraction=0.4 --conf spark.storage.memoryFraction=0.4 --master spark://`cat ~/spark-ec2/masters`:7077 --class Main ./neuronforest-spark.jar data_root=/mnt/data master= subvolumes=.*36 dimOffsets=0 malisGrad=100 nTrees=1 iterations=4

(cat ~/spark-ec2/slaves) | (tasks=""; while read line; do
	ssh -n -o StrictHostKeyChecking=no -t -t root@$line "source aws-credentials && /usr/local/aws/bin/aws s3 cp /masters_predictions/ s3://neuronforest.sparkdata/predictions --recursive" &
tasks="$tasks $!"; done; for t in $tasks; do wait $t; done)


(cat ~/spark-ec2/slaves) | (while read line; do
	ssh -n -o StrictHostKeyChecking=no -t -t root@$line "rm -rf /masters_predictions" &
tasks="$tasks $!"; done; for t in $tasks; do wait $t; done)