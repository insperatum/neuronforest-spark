#MASTER
	MASTER=`spark-ec2 -k luke -i ~/luke.pem --region=eu-west-1 get-master NAME | tail -1`


#download all data onto workers (don't do this any more)
(ssh -i ~/luke.pem root@$MASTER 'cat /root/spark-ec2/slaves' && echo $MASTER) | while read line; do
	ssh -n -o StrictHostKeyChecking=no -i ~/luke.pem -t -t root@$line 'source aws_credentials.sh && /root/.local/lib/aws/bin/aws s3 cp s3://neuronforest.sparkdata /mnt/data --recursive' &
done

#watch log for one worker
(ssh -i ~/luke.pem root@$MASTER 'tail -1 /root/spark-ec2/slaves') | while read line; do ssh -n -o StrictHostKeyChecking=no -i ~/luke.pem -t -t root@$line 'cd /root/spark/work/ && echo $(ls | tail -1) && cd $(ls | tail -1) && tail -f ./*/stdout' ; done

#watch stats for one worker
(ssh -i ~/luke.pem root@$MASTER 'tail -1 /root/spark-ec2/slaves') | while read line; do ssh -n -o StrictHostKeyChecking=no -i ~/luke.pem -t -t root@$line 'watch "df && echo && cat /proc/meminfo"' ; done


(ssh -i ~/luke.pem root@$MASTER 'cat /root/spark-ec2/slaves' && echo $MASTER) | while read line; do
	scp -o StrictHostKeyChecking=no -r -i ~/luke.pem root@$line:/masters_predictions/2014-12-29\ 02-29-39
/masters_predictions/ &
done


#clear existing predictions
(ssh -i ~/luke.pem root@$MASTER 'cat /root/spark-ec2/slaves' && echo $MASTER) | while read line; do
	ssh -n -o StrictHostKeyChecking=no -i ~/luke.pem -t -t root@$line 'rm -r /masters_predictions; mkdir /masters_predictions'
done


#Install aws CLI on a new cluster
(echo $MASTER && ssh -n -i ~/luke.pem root@$MASTER 'cat /root/spark-ec2/slaves') | (tasks=""; for v in ${volumes[0]} ${volumes[@]}; do
	read line; ssh -n -o StrictHostKeyChecking=no -i ~/luke.pem -t -t root@$line "curl 'https://s3.amazonaws.com/aws-cli/awscli-bundle.zip' -o 'awscli-bundle.zip' && yes | unzip awscli-bundle.zip && sudo yes | ./awscli-bundle/install -i /usr/local/aws -b /usr/local/bin/aws" &
tasks="$tasks $!"; done; for t in $tasks; do wait $t; done)



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

dt=$(date '+%Y%m%d_%H%M%S');
mkdir /root/logs
~/spark/bin/spark-submit --executor-memory 28G --driver-memory 122G --conf spark.shuffle.spill=false --conf spark.shuffle.memoryFraction=0.4 --conf spark.storage.memoryFraction=0.4 --master spark://`cat ~/spark-ec2/masters`:7077 --class Main ./neuronforest-spark.jar maxMemoryInMB=500 data_root=/mnt/data localDir=/mnt/tmp master= subvolumes=.*36 dimOffsets=0 malisGrad=100 initialTrees=50 treesPerIteration=0 iterations=0 maxDepth=15 testPartialModels=0,9,49,99 testDepths=5,10,15 > "/root/logs/$dt stdout.txt" 2> "/root/logs/$dt stderr.txt" &&
(cat ~/spark-ec2/slaves) | (tasks=""; while read line; do
	ssh -n -o StrictHostKeyChecking=no -t -t root@$line "source aws-credentials && /usr/local/aws/bin/aws s3 cp /masters_predictions/ s3://neuronforest.sparkdata/predictions --recursive" &
 tasks="$tasks $!"; done; for t in $tasks; do wait $t; done) &&
(cat ~/spark-ec2/slaves) | (while read line; do
	ssh -n -o StrictHostKeyChecking=no -t -t root@$line "rm -rf /masters_predictions" &
tasks="$tasks $!"; done; for t in $tasks; do wait $t; done)




#Run (from home) and exit THIS DOESN'T WORK, DO I NEED A -t -t or something??
ssh -i ~/luke.pem root@$MASTER './experiment.sh'; spark-ec2 -k luke -i ~/luke.pem --region=eu-west-1 -s 1 stop BIG36 