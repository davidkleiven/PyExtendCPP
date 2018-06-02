prefix="/usr/local/include"

install:
	mkdir -p "${prefix}/pyextend"
	cp include/*.hpp "${prefix}/pyextend"
