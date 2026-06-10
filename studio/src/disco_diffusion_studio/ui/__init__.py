"""The presentation layer: the screen areas (sidebar, bottom bar, canvas) + render/event modules.

Each area class owns its widgets and its build / event-handling / sync behaviour; the App wires
them together and they reach shared state/actions back through the App they're handed.
"""
