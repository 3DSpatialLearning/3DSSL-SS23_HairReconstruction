#!
prog="./gipuma"
warping="../fusibile/fusibile"
inputdir="data/multiface2/"
orientaionMapsDir="data/multiface2/orient"
confidenceValuesDir="data/multiface2/conf"
masksDir="data/multiface2/mask"
batch_name="multiface2_with_conf_1_test_read_conf"
output_dir_basename="results/$batch_name"
p_folder="data/multiface2/cam_param.txt"
scale=1
blocksize=11
iter=8
cost_gamma=10
cost_comb="best_n"
n_best=3
depth_max=1300
depth_min=500
image_list_array=`( cd $inputdir && ls *.png) `
output_dir=${output_dir_basename}/

# fuse options
disp_thresh=1
normal_thresh=10
num_consistent=1
min_angle=1
max_angle=60

# strand accurate hair reconstruction params
k=21
rk=10

#warping conf
count=0
for im in $image_list_array
do
    echo $count
    img=${im%.png}
    cmd_file=${output_dir}/$img-cmd.log
    image_list=( $im )

    mkdir -p $output_dir
    for ij in $image_list_array
    do
	if [ $im != $ij ]
	then
	    image_list+=( $ij )
	fi
    done


    # cmd="$prog 25.png 3.png 10.png 11.png 12.png 14.png 15.png 16.png 18.png 19.png 21.png 22.png 23.png 24.png 28.png 27.png 29.png 2.png 30.png 1.png 4.png 5.png 6.png 7.png 8.png 9.png  
    cmd="$prog ${image_list[@]}
        -images_folder $inputdir
        -krt_file $p_folder
        -output_folder $output_dir
        -no_display
        --cam_scale=$scale
        --iterations=$iter
        --blocksize=$blocksize
        --cost_gamma=$cost_gamma
        --cost_comb=best_n
        --n_best=$n_best
        --depth_max=$depth_max
        --depth_min=$depth_min
        --min_angle=$min_angle
        --max_angle=$max_angle
        -k=$k
        -rk=$rk
        -orientation_maps_folder $orientaionMapsDir
        -confidence_values_folder $confidenceValuesDir
        -masks_folder $masksDir"
    echo $cmd
    $cmd

    let "count += 1"
    if [ $count -eq -1 ]
    then
	    break
    fi
done

echo $warping -input_folder $output_dir -krt_file $p_folder -images_folder $inputdir --cam_scale=$scale --depth_min=$depth_min --depth_max=$depth_max --disp_thresh=$disp_thresh --normal_thresh=$normal_thresh --num_consistent=$num_consistent --min_angle=$min_angle --max_angle=$max_angle
$warping -input_folder $output_dir -krt_file $p_folder -images_folder $inputdir --cam_scale=$scale --depth_min=$depth_min --depth_max=$depth_max --disp_thresh=$disp_thresh --normal_thresh=$normal_thresh --num_consistent=$num_consistent -remove_black_background --min_angle=$min_angle --max_angle=$max_angle
