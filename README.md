neuronforest-spark
==================

(uses spark commit id: e895e0cbecbbec1b412ff21321e57826d2d0a982)

**To setup on EC2:**

./spark-ec2 -k luke -i ~/luke.pem -s 36 --instance-type=r3.xlarge --master-instance-type=r3.4xlarge --region=eu-west-1 --spark-version=e895e0cbecbbec1b412ff21321e57826d2d0a982 launch *NAME*

MASTER=`spark-ec2 -k luke -i ~/luke.pem --region=eu-west-1 get-master *NAME* | tail -1`

(echo $MASTER && ssh -n -i ~/luke.pem root@$MASTER 'cat /root/spark-ec2/slaves') | (tasks=""; for v in ${volumes[0]} ${volumes[@]}; do
	read line; ssh -n -o StrictHostKeyChecking=no -i ~/luke.pem -t -t root@$line 'curl "https://s3.amazonaws.com/aws-cli/awscli-bundle.zip" -o "awscli-bundle.zip" && unzip awscli-bundle.zip && sudo ./awscli-bundle/install -i /usr/local/aws -b /usr/local/bin/aws' &
tasks="$tasks $!"; done; for t in $tasks; do wait $t; done)

./spark-ec2 -k luke -i ~/luke.pem --region=eu-west-1 login *NAME*

*export aws credentials*

spark-ec2/copy-dir aws-credentials
