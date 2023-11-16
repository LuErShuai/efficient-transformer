local dap = require('dap')
dap.adapters.python = {
    type = 'excutable',
    command = 'python',
    args = {'-m', 'debugpy.adapter'}
}

dap.configurations.python = {
    {
        type = 'python',
        request = 'launch',
        name = 'Launch file',
        program = '${file}'
    }
}
