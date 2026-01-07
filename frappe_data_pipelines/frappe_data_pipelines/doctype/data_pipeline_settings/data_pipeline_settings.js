// Data Pipeline Settings client-side script
frappe.ui.form.on('Data Pipeline Settings', {
    refresh: function(frm) {
        // Update dimension on load
        update_embedding_dimension(frm);

        // Setup button click handlers
        setup_action_buttons(frm);
    },

    embedding_provider: function(frm) {
        update_embedding_dimension(frm);
    },

    ollama_model: function(frm) {
        update_embedding_dimension(frm);
    },

    openrouter_model: function(frm) {
        update_embedding_dimension(frm);
    }
});

function update_embedding_dimension(frm) {
    const ollama_dimensions = {
        'nomic-embed-text': 768,
        'mxbai-embed-large': 1024,
        'all-minilm': 384,
        'snowflake-arctic-embed': 1024
    };

    const openrouter_dimensions = {
        'openai/text-embedding-3-small': 1536,
        'openai/text-embedding-3-large': 3072,
        'openai/text-embedding-ada-002': 1536,
        'cohere/embed-english-v3.0': 1024,
        'cohere/embed-multilingual-v3.0': 1024,
        'cohere/embed-english-light-v3.0': 384
    };

    let dimension = 768; // default

    if (frm.doc.embedding_provider === 'Local (Ollama)') {
        const model = frm.doc.ollama_model || 'nomic-embed-text';
        dimension = ollama_dimensions[model] || 768;
    } else if (frm.doc.embedding_provider === 'OpenRouter') {
        const model = frm.doc.openrouter_model || 'openai/text-embedding-3-small';
        dimension = openrouter_dimensions[model] || 1536;
    }

    frm.set_value('embedding_dimension', dimension);
}

function setup_action_buttons(frm) {
    // Wait for DOM to be ready
    setTimeout(function() {
        // Test Connections button
        $('#test-connections-btn').off('click').on('click', function() {
            test_connections(frm);
        });

        // Process Existing Files button
        $('#process-existing-btn').off('click').on('click', function() {
            process_existing_files(frm);
        });

        // Refresh Stats button
        $('#refresh-stats-btn').off('click').on('click', function() {
            refresh_stats(frm);
        });
    }, 100);
}

function test_connections(frm) {
    frappe.show_alert({message: 'Testing connections...', indicator: 'blue'});

    frappe.call({
        method: 'frappe_data_pipelines.frappe_data_pipelines.doctype.data_pipeline_settings.data_pipeline_settings.test_connections',
        callback: function(r) {
            if (r.message) {
                if (r.message.success) {
                    frappe.show_alert({message: 'All connections successful!', indicator: 'green'});
                    frm.set_value('connection_status', 'Connected');
                } else {
                    frappe.show_alert({message: 'Connection failed: ' + r.message.error, indicator: 'red'});
                    frm.set_value('connection_status', 'Failed: ' + r.message.error);
                }
            }
        },
        error: function(r) {
            frappe.show_alert({message: 'Connection test failed', indicator: 'red'});
        }
    });
}

function process_existing_files(frm) {
    frappe.confirm(
        'This will process all existing Drive files. Continue?',
        function() {
            frappe.show_alert({message: 'Starting batch processing...', indicator: 'blue'});

            frappe.call({
                method: 'frappe_data_pipelines.frappe_data_pipelines.doctype.data_pipeline_settings.data_pipeline_settings.process_existing_files',
                callback: function(r) {
                    if (r.message) {
                        frappe.show_alert({
                            message: 'Queued ' + r.message.queued + ' files for processing',
                            indicator: 'green'
                        });
                    }
                },
                error: function(r) {
                    frappe.show_alert({message: 'Failed to start processing', indicator: 'red'});
                }
            });
        }
    );
}

function refresh_stats(frm) {
    frappe.show_alert({message: 'Refreshing statistics...', indicator: 'blue'});

    frappe.call({
        method: 'frappe_data_pipelines.frappe_data_pipelines.doctype.data_pipeline_settings.data_pipeline_settings.get_pipeline_stats',
        callback: function(r) {
            if (r.message) {
                var stats = r.message;
                var html = '<strong>Pipeline Statistics:</strong><br>';
                html += '<ul style="margin-bottom: 0;">';
                html += '<li>Total Documents: ' + (stats.total_documents || 0) + '</li>';
                html += '<li>Total Chunks: ' + (stats.total_chunks || 0) + '</li>';
                html += '<li>Pending Jobs: ' + (stats.pending_jobs || 0) + '</li>';
                html += '<li>Failed Jobs: ' + (stats.failed_jobs || 0) + '</li>';
                html += '</ul>';
                $('#pipeline-stats-display').html(html);
                frappe.show_alert({message: 'Statistics refreshed', indicator: 'green'});
            }
        },
        error: function(r) {
            frappe.show_alert({message: 'Failed to refresh statistics', indicator: 'red'});
        }
    });
}
