"""annotate_spatial pipeline: images."""
from __future__ import annotations
from .common import SpatialLLMNode

def read_views(accessor, runner, llm, state, slides, question):
    """Render neutral views without an intermediate model interpretation."""
    import hashlib
    from .composition import ComputeSpatialProportionsNode
    from sclambda.annotation.niche.measurements import observation_record, identify
    renderer = ComputeSpatialProportionsNode(accessor, runner)
    records = []
    for slide in dict.fromkeys(slides):
        if state.get("slice_key") and slide is None:
            continue  # Never superimpose independent coordinate frames.
        images, legend = renderer._generate_image(state["cluster_id"], state.get("spatial_key", "spatial"),
            state.get("slice_key"), slide, image_px=state.get("spatial_image_px", 700),
            flip=state.get("flip_spatial_y", False), background_col=state.get("spatial_background_col"))
        if not images:
            continue
        views = [{"slide": slide, "mode": "whole" if i == 0 else "crop",
                  "flip_y": bool(state.get("flip_spatial_y", False)),
                  "coordinate_frame": "shared section origin and scale; not anatomical orientation",
                  "sha256": hashlib.sha256(im.encode()).hexdigest()} for i, im in enumerate(images)]
        value = observation_record(state["cluster_id"],
            f"Target niche {state['cluster_id']} is RED; every other observation is GRAY. "
            "Views are titled whole section followed by a crop containing the full target extent. "
            "The dashed box locates the crop; both views use the same section coordinate frame. "
            "All disconnected target locations are retained. Axes are plotting coordinates, "
            "not anatomical orientation. Inspect the actual red mask, not nearby populations. "
            f"View requested for: {question}",
            {"slice_key": state.get("slice_key"), "spatial_slide": slide}, views=views,
            kind="spatial_view")
        value["source"] = "dataset coordinates and target mask; rendering metadata, not an anatomical interpretation"
        value = identify(value)
        value["images"] = images  # Preserve original images, omitted from textual rendering.
        records.append(value)
    return records


class SpatialImageDescriptionNode(SpatialLLMNode):
    default_output = {"spatial_image_description": None, "spatial_observations": []}

    def execute(self, state):
        if not state.get("include_spatial_image", True):
            return self.output
        records = read_views(self.accessor, self.runner, self.llm_client, state,
            state.get("image_slides") or [state.get("selected_slide", state.get("slice"))],
            "Describe the niche's position on this section; identify uncertain observations.")
        self.output["spatial_observations"] = records
        self.output["spatial_image_description"] = "\n".join(
            f"Slide {m['scope']['spatial_slide']}: {m['rows'][0]['text']}" for m in records)
        return self.output
