#!
prog="./gipuma"
warping="../fusibile/fusibile"
inputdir="data/blender_line/"
batch_name="blender_line"
output_dir_basename="results/$batch_name"
p_folder="data/blender_line/cam_par.txt"
scale=1
blocksize=11
iter=8
cost_gamma=10
cost_comb="best_n"
n_best=3
depth_max=1.7
depth_min=0.7
image_list_array=`( cd $inputdir && ls *.png) `
output_dir=${output_dir_basename}/

# fuse options
disp_thresh=7
normal_thresh=70
num_consistent=3
min_angle=30
max_angle=90

# strand accurate hair reconstruction params
k=141
rk=100

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
    cmd="$prog ${image_list[@]} -images_folder $inputdir -krt_file $p_folder -output_folder $output_dir -no_display --cam_scale=$scale --iterations=$iter --blocksize=$blocksize --cost_gamma=$cost_gamma --cost_comb=best_n --n_best=$n_best --depth_max=$depth_max --depth_min=$depth_min --min_angle=$min_angle --max_angle=$max_angle -color_processing -k=$k -rk=$rk"
    echo $cmd
    $cmd

    
    let "count += 1"

    # Uncomment this line to run for multiple images
    let "count = -1"

    if [ $count -eq -1 ]
    then
	    break
    fi
done
