# Part-Aware Shape Generation






## Scorer

Generation of multiple view from Obj file.

Generating 12 views for Chairs in PartNet Using 6 processes

find -f ./Chair/models *.obj -print0 | xargs -0 -n1 -P6 -I {} Blender --background --python stanford-shapenet-renderer/render_blender.py -- --views 12 --output_folder ./ChairViews {}


Source:
https://github.com/panmari/stanford-shapenet-renderer

Example invocation

To render a single .obj file, run

blender --background --python render_blender.py -- --output_folder /tmp path_to_model.obj
To get raw values that are easiest for further use, use --format OPEN_EXR. If the .obj file references any materials defined in a .mtl file, it is assumed to be in the same folder with the same name.

Batch rendering

To render a whole batch, you can e. g. use the unix tool find:

find . -name *.obj -exec blender --background --python render_blender.py -- --output_folder /tmp {} \;
To speed up the process, you can also use xargs to have multiple blender instances run in parallel using the -P argument

find . -name *.obj -print0 | xargs -0 -n1 -P3 -I {} blender --background --python render_blender.py -- --output_folder /t
