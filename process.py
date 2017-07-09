import detect_lane
from moviepy.editor import VideoFileClip


def create_video(input_file, output_file):
    """
    Creates an output video with lane overlaid on a given video
    :param input_file: File that needs to be read
    :param output_file: File which will be created with lane lines overlaid
    :return: None
    """
    input_video = VideoFileClip(input_file)
    output_video = input_video.fl_image(detect_lane.fit_and_plot)
    output_video.write_videofile(output_file, audio=False)


create_video('project_video.mp4', 'project_video-output.mp4')
