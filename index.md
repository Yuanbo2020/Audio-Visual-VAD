<h1 align="center">Rule-embedded network for audio-visual voice activity detection in live musical video streams<p></p></h1>

<h2 align="center">Paper<p></p></h2>

Please see <a href="https://arxiv.org/abs/2010.14168" 
target="https://arxiv.org/abs/2010.14168">here</a> (https://arxiv.org/abs/2010.14168).<br>

<h2 align="center">Code<p></p></h2>

Please see <a href="https://github.com/Yuanbo2020/Audio-Visual-VAD/tree/main/Code" 
target="https://github.com/Yuanbo2020/Audio-Visual-VAD/tree/main/Code">here</a>.<br>

<h2 align="center">Open dataset MAVC100<p></p></h2>

For detailed information and download of the MAVC100, please see <a href="https://github.com/Yuanbo2020/Audio-Visual-VAD/tree/main/Open_dataset_MAVC100" 
target="https://github.com/Yuanbo2020/Audio-Visual-VAD/tree/main/Open_dataset_MAVC100">here</a>.<br>

<h2 align="center">Demos of the detection results<p></p></h2>

Here are the detection results based on rule-embedded audio-visual VAD network in the paper.

<br>

The font on the top left of the video shows the activity of the anchor at the current moment. The anchor speaks, it shows speech; the anchor sings, it shows singing; the anchor has no action and there is sound in the background, it shows silence; otherwise it shows others.

<h3 align="center">Demo 1<p></p></h3>
<div align="center">
<video width=18%/ controls>
<source src="https://github.com/Yuanbo2020/Audio-Visual-VAD/blob/main/Video_demos/demo2.mp4" type="video/mp4">   
</video>
</div>  

<h3 align="center">Demo 2<p></p></h3>
<div align="center">
<video width=18%/ controls>
<source src="https://github.com/Yuanbo2020/Audio-Visual-VAD/blob/main/Video_demos/demo3.mp4" type="video/mp4">   
</video>
</div>  

<h3 align="center">Demo 3<p></p></h3>
<div align="center">
<video width=18%/ controls>
<source src="https://github.com/Yuanbo2020/Audio-Visual-VAD/blob/main/Video_demos/demo1.mp4" type="video/mp4">   
</video>
</div>  

<h2 align="center">The proposed rule-embedded AV-VAD network<p></p></h2>

<p><div align="center">
<img src="https://github.com/Yuanbo2020/Audio-Visual-VAD/blob/main/The_proposed_rule-embedded_AV-VAD_network.png" width=100%/>
</div>

The left part is audio branch (red words) that tries to learn the high-level acoustic features of target events in audio level, and right part is image branch (blue words) attempts to judge whether the anchor is vocalizing using visual information. The bottom part is the Audio-Visual branch (purple italics), which aims to fuse the bi-modal representations to determine the probability of target events of this paper.

<h2 align="center">The original output of the rule-embedded AV-VAD network<p></p></h2>

<p><div align="center">
<img src="https://github.com/Yuanbo2020/Audio-Visual-VAD/blob/main/The_original_output_of_the_rule-embedded_AV-VAD_network.png" width=100%/>
</div>
</p>

<p align="center">In subgraph (a), the red, blue, gray and green lines denote the probability of Singing, Speech, Others and Silence in audio, respectively.<br>
In subgraph (b), the gray and black lines denote the probability of vocalizing and non-vocalizing, respectively.<br>
In subgraph (c), the red, blue and gray lines denote the probability of target Singing, Speech and Others, and the other remaining part is Silence.</p>



