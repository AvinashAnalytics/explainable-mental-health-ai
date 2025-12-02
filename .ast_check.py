import ast, sys
p='c:\\Users\\Avinash rai\\Downloads\\Major proj AWA Proj\\Major proj AWA\\train_depression_classifier_gpu.py'
with open(p,'r',encoding='utf-8') as f:
    src=f.read()
mod=ast.parse(src)
for node in mod.body:
    if isinstance(node, ast.FunctionDef) and node.name=='train_model':
        print('Found train_model')
        # walk nodes to find any assignment to name 'torch'
        for n in ast.walk(node):
            if isinstance(n, ast.Assign):
                for target in n.targets:
                    if isinstance(target, ast.Name) and target.id=='torch':
                        print('Assignment to torch found at line', n.lineno)
            if isinstance(n, ast.AnnAssign):
                target=n.target
                if isinstance(target, ast.Name) and target.id=='torch':
                    print('AnnAssign to torch at', n.lineno)
            if isinstance(n, ast.arguments):
                for arg in n.args:
                    if arg.arg=='torch':
                        print('Argument named torch in function at', node.lineno)
            if isinstance(n, ast.ExceptHandler):
                if n.name=='torch':
                    print('Except as torch at', n.lineno)
print('Done')
