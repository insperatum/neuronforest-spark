NAME=BIG36

#Launch
spark-ec2 -k luke -i ~/luke.pem --region=eu-west-1 --instance-type=r3.xlarge start $NAME

#MASTER
MASTER=`spark-ec2 -k luke -i ~/luke.pem --region=eu-west-1 get-master $NAME | tail -1`

#####FOR THE FIRST TIME#####
scp -i ~/luke.pem /home/luke/aws-credentials root@$MASTER: && ssh -i ~/luke.pem root@$MASTER '~/spark-ec2/copy-dir aws-credentials'

(echo $MASTER && ssh -n -i ~/luke.pem root@$MASTER 'cat /root/spark-ec2/slaves') | (tasks=""; while read line; do ssh -n -o StrictHostKeyChecking=no -i ~/luke.pem -t -t root@$line 'curl "https://s3.amazonaws.com/aws-cli/awscli-bundle.zip" -o "awscli-bundle.zip" && unzip awscli-bundle.zip && sudo ./awscli-bundle/install -i /usr/local/aws -b /usr/local/bin/aws' & tasks="$tasks $!"; done; for t in $tasks; do wait $t; done)

ssh -i ~/luke.pem root@$MASTER 'git clone http://github.com/insperatum/scripts.git'
############################

#download *ALL* data onto workers (for isbi)
(ssh -i ~/luke.pem root@$MASTER 'cat /root/spark-ec2/slaves' && echo $MASTER) | (while read line; do
	ssh -n -o StrictHostKeyChecking=no -i ~/luke.pem -t -t root@$line 'source aws-credentials && aws s3 sync s3://neuronforest.sparkdata/isbi_data /mnt/data' & tasks="$tasks $!"; done; for t in $tasks; do wait $t; done);

#Copy application over
scp -i ~/luke.pem /home/luke/IdeaProjects/neuronforest-spark/out/artifacts/neuronforest-spark.jar root@$MASTER:

#login
spark-ec2 -k luke -i ~/luke.pem --region=eu-west-1 login $NAME

#!!!!!! DO SOMETHING TO FIX THE /mnt2/spark thing !!!!!!
#!!!!!! sbin/stop-all -> sbin/start-all !!!!!!
#!!!!!! or start with correct instance type?? !!!!!!




###WATCHING
#view latest logs
export latest="logs/`ls logs | tail -n 1 | awk '{print $1;}'` stdout.txt"; let "lines=`grep -n getting "$latest" | cut -f1 -d:`-1"; head -n$lines "$latest" ; echo ""; echo "..."; echo ""; tail -f "$latest"



#STOP 
yes | spark-ec2 -i ~/luke.pem --region eu-west-1 stop BIG36














# -------------- OLD SHIT -------------------
#watch log for one worker
(ssh -i ~/luke.pem root@$MASTER 'tail -1 /root/spark-ec2/slaves') | while read line; do ssh -n -o StrictHostKeyChecking=no -i ~/luke.pem -t -t root@$line 'cd /root/spark/work/ && echo $(ls | tail -1) && cd $(ls | tail -1) && tail -f ./*/stdout' ; done

#watch stats for one worker
(ssh -i ~/luke.pem root@$MASTER 'tail -1 /root/spark-ec2/slaves') | while read line; do ssh -n -o StrictHostKeyChecking=no -i ~/luke.pem -t -t root@$line 'watch "df && echo && cat /proc/meminfo"' ; done


(ssh -i ~/luke.pem root@$MASTER 'cat /root/spark-ec2/slaves' && echo $MASTER) | while read line; do
	scp -o StrictHostKeyChecking=no -r -i ~/luke.pem root@$line:/masters_predictions/2014-12-29\ 02-29-39
/masters_predictions/ &
done


#download *different* data to each worker
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