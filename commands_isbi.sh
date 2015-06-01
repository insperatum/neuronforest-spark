#MASTER
	MASTER=`spark-ec2 -k luke -i ~/luke.pem --region=eu-west-1 get-master BIG36 | tail -1`

#download all data onto workers
(ssh -i ~/luke.pem root@$MASTER 'cat /root/spark-ec2/slaves' && echo $MASTER) | while read line; do
	ssh -n -o StrictHostKeyChecking=no -i ~/luke.pem -t -t root@$line 'source aws-credentials && aws s3 cp s3://neuronforest.sparkdata/isbi_data /mnt/isbi_data --recursive' &
done

#Copy application over
scp -i ~/luke.pem /home/luke/IdeaProjects/neuronforest-spark/out/artifacts/neuronforest-spark.jar root@$MASTER:

#login
spark-ec2 -k luke -i ~/luke.pem --region=eu-west-1 login BIG36



#FILES

#clear
rm -rf /mnt/isbi_models; rm -rf /mnt/isbi_predictions;
(cat ~/spark-ec2/slaves) | (while read line; do
 ssh -n -o StrictHostKeyChecking=no -t -t root@$line "rm -rf /mnt/isbi_predictions" & 
 tasks="$tasks $!"; done; for t in $tasks; do wait $t; done);
mkdir /root/logs

#clearlogs
rm -rf ~/logs/*
(cat ~/spark-ec2/slaves) | (while read line; do
 ssh -n -o StrictHostKeyChecking=no -t -t root@$line "rm -rf ~/spark/work/*" & 
 tasks="$tasks $!"; done; for t in $tasks; do wait $t; done);


#subvolumes
export SUBVOLUMES_TRAIN_ONE=r1f0s0
export SUBVOLUMES_TRAIN_MINI=r1f0s0,r1f0s1,r1f0s2,r1f0s3,r1f0s4
export SUBVOLUMES_TRAIN=r1f0s0,r1f0s1,r1f0s10,r1f0s11,r1f0s12,r1f0s13,r1f0s14,r1f0s15,r1f0s16,r1f0s17,r1f0s18,r1f0s19,r1f0s2,r1f0s20,r1f0s21,r1f0s22,r1f0s23,r1f0s24,r1f0s25,r1f0s26,r1f0s27,r1f0s28,r1f0s29,r1f0s3,r1f0s4,r1f0s5,r1f0s6,r1f0s7,r1f0s8,r1f0s9
export SUBVOLUMES_TRAIN_ROTATIONS=r1f0s0,r1f0s1,r1f0s10,r1f0s11,r1f0s12,r1f0s13,r1f0s14,r1f0s15,r1f0s16,r1f0s17,r1f0s18,r1f0s19,r1f0s2,r1f0s20,r1f0s21,r1f0s22,r1f0s23,r1f0s24,r1f0s25,r1f0s26,r1f0s27,r1f0s28,r1f0s29,r1f0s3,r1f0s4,r1f0s5,r1f0s6,r1f0s7,r1f0s8,r1f0s9,r2f0s0,r2f0s1,r2f0s10,r2f0s11,r2f0s12,r2f0s13,r2f0s14,r2f0s15,r2f0s16,r2f0s17,r2f0s18,r2f0s19,r2f0s2,r2f0s20,r2f0s21,r2f0s22,r2f0s23,r2f0s24,r2f0s25,r2f0s26,r2f0s27,r2f0s28,r2f0s29,r2f0s3,r2f0s4,r2f0s5,r2f0s6,r2f0s7,r2f0s8,r2f0s9,r3f0s0,r3f0s1,r3f0s10,r3f0s11,r3f0s12,r3f0s13,r3f0s14,r3f0s15,r3f0s16,r3f0s17,r3f0s18,r3f0s19,r3f0s2,r3f0s20,r3f0s21,r3f0s22,r3f0s23,r3f0s24,r3f0s25,r3f0s26,r3f0s27,r3f0s28,r3f0s29,r3f0s3,r3f0s4,r3f0s5,r3f0s6,r3f0s7,r3f0s8,r3f0s9
export SUBVOLUMES_TRAIN_FULL=r1f0s0,r1f0s1,r1f0s10,r1f0s11,r1f0s12,r1f0s13,r1f0s14,r1f0s15,r1f0s16,r1f0s17,r1f0s18,r1f0s19,r1f0s2,r1f0s20,r1f0s21,r1f0s22,r1f0s23,r1f0s24,r1f0s25,r1f0s26,r1f0s27,r1f0s28,r1f0s29,r1f0s3,r1f0s4,r1f0s5,r1f0s6,r1f0s7,r1f0s8,r1f0s9,r1f1s0,r1f1s1,r1f1s10,r1f1s11,r1f1s12,r1f1s13,r1f1s14,r1f1s15,r1f1s16,r1f1s17,r1f1s18,r1f1s19,r1f1s2,r1f1s20,r1f1s21,r1f1s22,r1f1s23,r1f1s24,r1f1s25,r1f1s26,r1f1s27,r1f1s28,r1f1s29,r1f1s3,r1f1s4,r1f1s5,r1f1s6,r1f1s7,r1f1s8,r1f1s9,r2f0s0,r2f0s1,r2f0s10,r2f0s11,r2f0s12,r2f0s13,r2f0s14,r2f0s15,r2f0s16,r2f0s17,r2f0s18,r2f0s19,r2f0s2,r2f0s20,r2f0s21,r2f0s22,r2f0s23,r2f0s24,r2f0s25,r2f0s26,r2f0s27,r2f0s28,r2f0s29,r2f0s3,r2f0s4,r2f0s5,r2f0s6,r2f0s7,r2f0s8,r2f0s9,r2f1s0,r2f1s1,r2f1s10,r2f1s11,r2f1s12,r2f1s13,r2f1s14,r2f1s15,r2f1s16,r2f1s17,r2f1s18,r2f1s19,r2f1s2,r2f1s20,r2f1s21,r2f1s22,r2f1s23,r2f1s24,r2f1s25,r2f1s26,r2f1s27,r2f1s28,r2f1s29,r2f1s3,r2f1s4,r2f1s5,r2f1s6,r2f1s7,r2f1s8,r2f1s9,r3f0s0,r3f0s1,r3f0s10,r3f0s11,r3f0s12,r3f0s13,r3f0s14,r3f0s15,r3f0s16,r3f0s17,r3f0s18,r3f0s19,r3f0s2,r3f0s20,r3f0s21,r3f0s22,r3f0s23,r3f0s24,r3f0s25,r3f0s26,r3f0s27,r3f0s28,r3f0s29,r3f0s3,r3f0s4,r3f0s5,r3f0s6,r3f0s7,r3f0s8,r3f0s9,r3f1s0,r3f1s1,r3f1s10,r3f1s11,r3f1s12,r3f1s13,r3f1s14,r3f1s15,r3f1s16,r3f1s17,r3f1s18,r3f1s19,r3f1s2,r3f1s20,r3f1s21,r3f1s22,r3f1s23,r3f1s24,r3f1s25,r3f1s26,r3f1s27,r3f1s28,r3f1s29,r3f1s3,r3f1s4,r3f1s5,r3f1s6,r3f1s7,r3f1s8,r3f1s9
export SUBVOLUMES_TEST=r0f0s0,r0f0s1,r0f0s10,r0f0s11,r0f0s12,r0f0s13,r0f0s14,r0f0s15,r0f0s16,r0f0s17,r0f0s18,r0f0s19,r0f0s2,r0f0s20,r0f0s21,r0f0s22,r0f0s23,r0f0s24,r0f0s25,r0f0s26,r0f0s27,r0f0s28,r0f0s29,r0f0s3,r0f0s4,r0f0s5,r0f0s6,r0f0s7,r0f0s8,r0f0s9

#init
if [ -z "$4" ]; then echo "Please give 4 arguments!"; exit 1; fi
source ~/isbi/subvolumes.sh
~/isbi/clear.sh;
dt=$(date '+%Y%m%d_%H%M%S');
~/spark/bin/spark-submit --executor-memory 28G --driver-memory 120G --conf spark.shuffle.spill=false --conf spark.shuffle.memoryFraction=0.1 --conf spark.storage.memoryFraction=0.7 --master spark://`cat ~/spark-ec2/masters`:7077 --class Main ./neuronforest-spark.jar numExecutors=36 maxMemoryInMB=2500 data_root=/mnt/isbi_data localDir=/mnt/tmp master= subvolumes_train=$1 subvolumes_test=$SUBVOLUMES_TEST dimOffsets=$2 learningRate=1 initialTrees=$3 save_to=/mnt/isbi_predictions save_model_to=/mnt/isbi_models treesPerIteration=10 iterations=0 maxDepth=$4 testPartialModels=1 testDepths=$4 useNodeIdCache=false subsampleProportion=1 momentum=0 > "/root/logs/$dt stdout.txt" 2> "/root/logs/$dt stderr.txt" &&
~/isbi/save.sh


#save
(cat ~/spark-ec2/slaves) | (tasks=""; while read line; do
 ssh -n -o StrictHostKeyChecking=no -t -t root@$line "source aws-credentials && aws s3 cp /mnt/isbi_predictions/ s3://neuronforest.sparkdata/isbi_predictions --recursive" &
       tasks="$tasks $!"; done; for t in $tasks; do wait $t; done);
source aws-credentials && aws s3 cp /mnt/isbi_predictions/ s3://neuronforest.sparkdata/isbi_predictions --recursive
source aws-credentials && aws s3 cp /mnt/isbi_models/ s3://neuronforest.sparkdata/isbi_models --recursive



#load model
source ~/isbi/subvolumes.sh
~/isbi/clear.sh;
dt=$(date '+%Y%m%d_%H%M%S');
~/spark/bin/spark-submit --executor-memory 28G --driver-memory 120G --conf spark.shuffle.spill=false --conf spark.shuffle.memoryFraction=0.1 --conf spark.storage.memoryFraction=0.7 --master spark://`cat ~/spark-ec2/masters`:7077 --class Main ./neuronforest-spark.jar numExecutors=36 maxMemoryInMB=2500 data_root=/mnt/isbi_data localDir=/mnt/tmp master= subvolumes_train=$SUBVOLUMES_TRAIN_FULL subvolumes_test=$SUBVOLUMES_TEST dimOffsets=-0 learningRate=1 loadModel="/mnt/initial2" save_to=/mnt/isbi_predictions save_model_to=/mnt/isbi_models treesPerIteration=10 iterations=10 maxDepth=15 testPartialModels=100 testDepths=2,4,6,8,10,12,14,16,18,20,22,24,26,28,30 useNodeIdCache=false subsampleProportion=1 momentum=0 > "/root/logs/$dt stdout.txt" 2> "/root/logs/$dt stderr.txt" &&
~/isbi/save.sh

###WATCHING
#head and tail
export latest="logs/`ls logs | tail -n 1 | awk '{print $1;}'` stdout.txt"; head -n27 "$latest" && echo "---------------" && tail -f "$latest"
















# -------------- OLD SHIT
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