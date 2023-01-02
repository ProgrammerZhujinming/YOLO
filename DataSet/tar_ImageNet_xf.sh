for tar_file in ./ImageNet2012/*.tar;
do
    mkdir ${tar_file:0:24}
    tar xvf ${tar_file} -C ${tar_file:0:24}
done