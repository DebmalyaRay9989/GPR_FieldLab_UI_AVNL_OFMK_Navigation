# Navigation callback fix

Fixed the Dash `InvalidCallbackReturnValue` error triggered after login:

`Expected 0, got 4` for pattern-matching navigation outputs.

The top workspace chips now have stable `nav-top` IDs, both navigation surfaces update `current-page`, and the CSS-class callback derives its return lengths from `ctx.outputs_list`. Therefore, when a dynamic layout temporarily has zero matching components, it returns zero values instead of a fixed list of four.

Restart the server after replacing the package contents and hard-refresh the browser (Ctrl+F5).
