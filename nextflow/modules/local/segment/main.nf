process SEGMENT {
    tag "$meta.id"
    label 'process_high'
    label 'process_gpu'

    container "docker.io/${params.container_registry}/hne-nuclear-segmentation:${params.hne_nuclear_segmentation_version}"

    input:
    tuple val(meta), path(slide_path)

    output:
    tuple val(meta), path("${meta.id}")                                                 , emit: results
    tuple val(meta), path("${meta.id}/tiles.parquet")                                   , optional: true, emit: tiles
    tuple val(meta), path("${meta.id}/nuclei_stardist.parquet")                         , optional: true, emit: nuclei_stardist
    tuple val(meta), path("${meta.id}/nuclei_cellpose.parquet")                         , optional: true, emit: nuclei_cellpose
    tuple val(meta), path("${meta.id}/consensus_union.parquet")                         , optional: true, emit: consensus_union
    tuple val(meta), path("${meta.id}/consensus_intersection.parquet")                  , optional: true, emit: consensus_intersection
    tuple val(meta), path("${meta.id}/visualization.html")                              , optional: true, emit: viz_html
    tuple val(meta), path("${meta.id}/details.pdf")                                     , optional: true, emit: viz_pdf
    path "versions.yml"                                                                 , emit: versions

    script:
    def args = task.ext.args ?: ''
    def gpu_flag = params.gpu ? '--gpu' : '--cpu'
    def viz_html_flag = params.viz_html ? '--viz-html' : '--no-viz-html'
    def viz_pdf_flag = params.viz_pdf ? '--viz-pdf' : '--no-viz-pdf'
    def mpp_override = params.mpp_override ? "--mpp ${params.mpp_override}" : ''
    """
    hne-segment run \\
        ${slide_path} \\
        --out-dir ${meta.id} \\
        --target-mpp ${params.target_mpp} \\
        --tile-size ${params.tile_size} \\
        --overlap ${params.overlap} \\
        --edge-fraction ${params.edge_fraction} \\
        --batch-size ${params.batch_size} \\
        --iou-threshold ${params.iou_threshold} \\
        --union-containment ${params.union_containment} \\
        ${gpu_flag} \\
        ${viz_html_flag} \\
        ${viz_pdf_flag} \\
        --viz-pdf-tiles ${params.viz_pdf_tiles} \\
        ${mpp_override} \\
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        hne-nuclear-segmentation: \$(python3 -c "import hne_nuclear_segmentation as m; print(getattr(m, '__version__', 'unknown'))")
    END_VERSIONS
    """
}
