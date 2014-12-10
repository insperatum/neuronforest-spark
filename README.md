neuronforest-spark
==================

(uses spark commit id: e895e0cbecbbec1b412ff21321e57826d2d0a982)

**To run on EC2:**

./spark-ec2 -k luke -i ~/luke.pem -s 8 --instance-type=m3.medium --region=eu-west-1 --spark-version=e895e0cbecbbec1b412ff21321e57826d2d0a982 launch *NAME*

scp -i ~/luke.pem /home/luke/neuronforest-spark/out/artifacts/neuronforest.jar root@*INSTANCE-ADDRESS*.compute.amazonaws.com:

./spark-ec2 -k luke -i ~/luke.pem --region=eu-west-1 login *NAME*

curl "https://s3.amazonaws.com/aws-cli/awscli-bundle.zip" -o "awscli-bundle.zip"

unzip awscli-bundle.zip

./awscli-bundle/install -b ~/aws

*export aws credentials*

~/aws s3 cp s3://neuronforest.sparkdata/spark sparkdata --recursive

./spark-ec2/copy-dir /root/sparkdata

./spark-ec2/copy-dir ./neuronforest.jar

./spark/bin/spark-submit --master spark://*local master address*:7077 --class Main ./neuronforest.jar data_root=/root/data/im1/split_2 master=


to run a command on all slaves:

cat ./spark-ec2/slaves | while read line; do ssh -t -t root@$line 'SOME COMMAND' & sleep 0.5; done


to stop:

./spark-ec2 -i luke.pem --region eu-west-1 stop *NAME*