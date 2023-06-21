#!
prog="./gipuma"
warping="../fusibile/fusibile"
inputdir="data/97_frame_00005/"
batch_name="97_frame_00005_cam"
output_dir_basename="results/$batch_name"
p_folder="data/97_frame_00005/cam_par.txt"
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
k=41
rk=10

# Temporary run Gipuma only once
cmd="./gipuma cam_222200045.png cam_220700191.png cam_221501007.png cam_222200036.png cam_222200037.png cam_222200038.png cam_222200039.png cam_222200040.png cam_222200041.png cam_222200042.png cam_222200043.png cam_222200044.png cam_222200046.png cam_222200047.png cam_222200048.png cam_222200049.png
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
    -color_processing
    -k=$k
    -rk=$rk"
echo $cmd
$cmd


# Runs gipuma for each image 
# #warping conf
# count=0
# for im in $image_list_array
# do
#     echo $count
#     img=${im%.png}
#     cmd_file=${output_dir}/$img-cmd.log
#     image_list=( $im )

#     mkdir -p $output_dir
#     for ij in $image_list_array
#     do
# 	if [ $im != $ij ]
# 	then
# 	    image_list+=( $ij )
# 	fi
#     done
#     cmd="$prog ${image_list[@]} -images_folder $inputdir -krt_file $p_folder -output_folder $output_dir -no_display --cam_scale=$scale --iterations=$iter --blocksize=$blocksize --cost_gamma=$cost_gamma --cost_comb=best_n --n_best=$n_best --depth_max=$depth_max --depth_min=$depth_min --min_angle=$min_angle --max_angle=$max_angle -color_processing -k=$k -rk=$rk"
#     echo $cmd
#     $cmd

    
#     let "count += 1"

#     if [ $count -eq -1 ]
#     then
# 	    break
#     fi
# done
