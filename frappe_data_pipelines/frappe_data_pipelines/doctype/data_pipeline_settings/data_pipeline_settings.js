// Data Pipeline Settings client-side script
frappe.ui.form.on('Data Pipeline Settings', {
    refresh: function(frm) {
        // Update dimension on load
        update_embedding_dimension(frm);
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
