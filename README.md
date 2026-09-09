# distilling-trusted-monitor-diversity

Can inference-time monitor diversity be distilled into a single trusted monitor
— recovering most of a 3-monitor ensemble's backdoor-detection gain at 1x
inference cost, using no labels and no stronger teacher?

Two distillation methods compared side by side: DPO on MACA-style
consensus-derived preference pairs, and supervised distillation from the
ensemble's mean scores. Evaluated as static classification on the
[ControlArena APPS backdoor dataset](https://huggingface.co/datasets/RoganInglis/apps-control-arena).

See `project-plan.md` for the full design, `CONTEXT.md` for vocabulary, and
`docs/adr/` for the decisions and their trade-offs.
