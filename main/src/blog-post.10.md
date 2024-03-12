# What is Container?

## Linux Man Page Numbering
- 1 : User Commands
- 2 : System Calls
- 3 : Library Functions
- 4 : Special Files
- 5 : File Formats and Conventions
- 6 : Games et al.
- 7 : Miscellanea
- 8 : System Administration tools and Daemons

## User Mode and Kernel Mode
- User Mode : When a user-mode program requests to run, a process and virtual address space (address space for that process) are created for it by OS. 
User-mode programs are less privileged than user-mode applications and are not allowed to access the system resources directly. for instance, if an application under user mode wants to access system resources, it will have to first go through the Operating system kernel by using syscalls.
- Kernel Mode : the kernel is the core program on which all the other operating system components rely, it is used to access the hardware components and schedule which processes should run on a computer system and when, and it also manages the application software and hardware interaction. 
hence it is the most privileged program, unlike other programs, it can directly interact with the hardware. when programs running under user mode need hardware access for example webcam, then first it has to go through the kernel by using a syscall, and to carry out these requests the CPU switches from user mode to kernel mode at the time of execution. 
after finally completing the execution of the process the CPU again switches back to the user mode.

## System Calls
- a system call is a routine that allows a user application to request actions that require special privileges. adding system calls is one of several ways to extend the functions provided by the kernel.
- when to call system calls? : 
  - read and write from files.
  - create or delete files.
  - create and manage new processes.
  - send and receive packets, through network connections.
  - access hardware devices.
  - etc..
![blog10_system_call_1.png](images%2Fblog10_system_call_1.png)
- system call execution process
  - 1: the processor executes a process in the user mode until a system call interrupts it.
  - 2: then on a priority basis, the system call is executed in the kernel mode.
  - 3: after the completion of system call execution, control returns to user mode.,
  - 4: the execution resumes in Kernel mode.
![blog10_system_call_2.png](images%2Fblog10_system_call_2.png)
### process vs thread
 - A process is the execution of a program. It includes the program itself, data, resources such as files, and execution info such as process relation information kept by the OS. 
The OS allows users to create, schedule, and terminate the processes via system calls.
 - A thread is a semi-process. It has its own stack and executes a given piece of code. Unlike a real process, the thread normally shares its memory with other threads. 
Conversely, processes usually have a different memory area for each one of them.
 - https://www.baeldung.com/cs/threads-sharing-resources

### fork, clone, exec, unshare, setns, nsenter and mount
- fork(2) : fork creates a new ("child") process by duplicating the calling process.
The new process is referred to as the child process. The calling process is referred to as the parent process. 
The child process and the parent process run in separate memory spaces.
The entire virtual address space of the parent is replicated in the child, including the states of mutexes, condition variables, and other pthreads objects.

- clone(2) : These system calls create a new ("child") process, in a manner similar to fork(2). 
By contrast with ``fork(2)``, these system calls provide more precise control over what pieces of execution context are shared between the calling process and the child process.  
For example, using these system calls, the caller can control whether or not the two processes share the virtual address space, the table of file descriptors, and the table of signal handlers.  
These system calls also allow the new child process to be placed in separate namespaces.

- execve(2) : The exec() family of functions replaces the current process image with a new process image. 
``execve()`` executes the program referred to by pathname.  This causes the program that is currently being run by the calling process to be replaced with a new program, with newly initialized stack, heap, and (initialized and uninitialized) data segments.
exec함수는 사용하던 공간을 해제하고 새로운 공간을 실행 가능한 바이너리를 로드하고 실행한다. 새로운 스택을 초기화하고 excutable entry point를 전달하여 실행한다. 
exec 함수는 보통 fork-and-exec 방식(UNIX process management model에서 일컫는)으로 사용된다. 어떤 프로세스가 exec를 호출하면 이 프로세스 메모리 공간을 해제하고 새 실행 파일을 로드하고 실행한다. 이렇게 되어버리면 exec를 호출한 프로세스 공간은 새 프로세스로 교체되고, exec를 호출한 프로세스는 실행이 중지된다. 
기존 프로세스의 실행 중지를 회피하기 위한 방법이 fork-and-exec 방식인 것이다. 새로운 프로세스를 생성하고 그 프로세스에서 exec 함수를 실행해서 새로운 프로세스를 실행하는 것이다. 이 방식을 사용하여 부모 프로세스와 자식 프로세스가 모두 실행이 가능한 것이다.

- unshare(1) : ``unshare(1)`` command creates new namespaces (as specified by the command-line options) and then executes the specified program.
  - mount namespace, UTS namespace, IPC namespace, PID namespace, network namespace, user namespace, cgroup namespace.
  - ``--mount-proc`` : just before running the program, mount the proc filesystem at mountpoint (default is /proc). This is useful when creating a new PID namespace. 
  It also implies creating a new mount namespace since the /proc mount would otherwise mess up existing programs on the system. 
  The new proc filesystem is explicitly mounted as private to make sure that the new namespace is really unshared.
  - ``--fork`` : fork the specified program as a child process of ``unshare`` rather than running it directly. This is useful when creating a new PID namespace. (``fork-and-exec`` implementing model)
  ``unshare`` program will fork a 'new process' after it has created the namespace. Then this 'new process' will have PID=1 and will execute our shell program. (``exec``)

- setns(2) : setns() system call can be used to associate the calling thread with a namespace of the specified type.
  - `` int setns(int fd, int nstype);`` : the ``fd argument is a file descriptor referring to one of the magic links in a "/proc/pid/ns/" directory (or a bind mount to such a link) or a PID file descriptor.
- nsenter(1) : run program in different namespaces.
- mount(8) : mount a filesystem. All files accessible in a Unix system are arranged in one big tree, the file hierarchy, rooted at ``/``. 
These files can be spread out over several devices. The mount command serves to attach the filesystem found on some device to the big file tree.
  - The ``proc`` filesystem is a pseudo-filesystem which provides an interface to kernel data structures.  It is commonly mounted at ``/proc``.  
  - Typically, it is mounted automatically by the system, but it can also be mounted manually using a command ```mount -t proc proc /proc```.
  - bind-mount : Classically, mounting creates a view of a storage device as a directory tree. A bind mount instead takes an existing directory tree and replicates it under a different point.

## CGroups
### concept
 - cgroups have 4 main features(resource limiting, prioritization, accounting, process control) to allow an administrator to ensure that programs running on the system stay within certain acceptable boundaries for CPU, RAM, block device I/O, and device groups.
 - cgroups are simply a directory structure with cgroups mounted into them. They can be located anywhere on the filesystem, but you will find the system-created cgroups in "/sys/fs/cgroup" by default.
#### terminology
- cgroup : collection of processes that are bound by the same set of limits or parameters defined via the cgroup filesystem.
- subsystem (= resource controller) :controller is a kernel component that modifies the behavior of the processes in a cgroup.  Various subsystems have been implemented, 
making it possible to do things such as limiting the amount of CPU time and memory available to a cgroup, accounting for the CPU time used by a cgroup, and freezing and resuming execution of the processes in a cgroup.
- hierarch : hierarchy is defined by creating, removing, and renaming subdirectories within the cgroup filesystem.  At each level of the hierarchy, attributes (e.g., limits) can be defined.

### implementation
![blog10_cgroup_diagram.png](images%2Fblog10_cgroup_diagram.png)
- "/sys/fs/cgroup" is the default mount point for cgroups.
- each type of controller (cpu, disk, memory, etc.) is subdivided into a tree-like structure like a filesystem.
- ``PID 1`` in memory, disk i/o, and cpu control groups. but cgroups are created per resource type and have no association with each other. so ``PID 1`` in the memory controller can actually has no relation to ``PID 1`` in the cpu controller.
```bash
# ls /sys/fs/cgroup (hierarchy)
/cgroup
├── <controller type>
│   ├── <group 1>
│   ├── <group 2>
│   ├── <group 3>

# mount hierarchy to cgroup controller
mount -t cgroup -o memory none /my_cgroups/memory
mount -t cgroup -o cpu,cpuacct none /my_cgroups/cpu
mount -t cgroup -o cpuset none /my_cgroups/cpusets

# ls -l /sys/fs/cgroup/cpu/user1/
-rw-r--r--. 1 root root 0 Sep  5 10:26 cgroup.clone_children
-rw-r--r--. 1 root root 0 Sep  5 10:26 cgroup.procs
-r--r--r--. 1 root root 0 Sep  5 10:26 cpuacct.stat
-rw-r--r--. 1 root root 0 Sep  5 10:26 cpuacct.usage
-r--r--r--. 1 root root 0 Sep  5 10:26 cpuacct.usage_all
-r--r--r--. 1 root root 0 Sep  5 10:26 cpuacct.usage_percpu
-r--r--r--. 1 root root 0 Sep  5 10:26 cpuacct.usage_percpu_sys
-r--r--r--. 1 root root 0 Sep  5 10:26 cpuacct.usage_percpu_user
-r--r--r--. 1 root root 0 Sep  5 10:26 cpuacct.usage_sys
-r--r--r--. 1 root root 0 Sep  5 10:26 cpuacct.usage_user
-rw-r--r--. 1 root root 0 Sep  5 10:26 cpu.cfs_period_us
-rw-r--r--. 1 root root 0 Sep  5 10:26 cpu.cfs_quota_us
-rw-r--r--. 1 root root 0 Sep  5 10:26 cpu.rt_period_us
-rw-r--r--. 1 root root 0 Sep  5 10:26 cpu.rt_runtime_us
-rw-r--r--. 1 root root 0 Sep  5 10:20 cpu.shares
-r--r--r--. 1 root root 0 Sep  5 10:26 cpu.stat
-rw-r--r--. 1 root root 0 Sep  5 10:26 notify_on_release
-rw-r--r--. 1 root root 0 Sep  5 10:23 tasks

# control the cpu shares with cgroup
$ echo 2048 > user1/cpu.shares
$ echo 768 > user2/cpu.shares
$ echo 512 > user3/cpu.shares

# add PID to cgroup 'user1'
$ echo 2023 > user1/tasks
```
### conclusion
![blog10_cgroup_conclusion.png](images%2Fblog10_cgroup_conclusion.png)

## Namespaces
### concept
- provide processes with their own system view, thus isolating independent processes from each other. 
- in other words, namespaces define the set of resources that a process can use (You cannot interact with something that you cannot see).

### implementation
- "/proc/[pid]/ns" is the default mount point for namespaces, and it contains symbolic links to the namespace files for each type of namespace that the process belongs to.
```bash
# ls -l /proc/5151/ns
total 0
lrwxrwxrwx 1 abdulhameed abdulhameed 0 Jul 26 06:48 cgroup -> 'cgroup:[4026531835]'
lrwxrwxrwx 1 abdulhameed abdulhameed 0 Jul 26 06:55 ipc -> 'ipc:[4026531839]'
lrwxrwxrwx 1 abdulhameed abdulhameed 0 Jul 26 06:55 mnt -> 'mnt:[4026531841]'
lrwxrwxrwx 1 abdulhameed abdulhameed 0 Jul 26 06:55 net -> 'net:[4026531840]'
lrwxrwxrwx 1 abdulhameed abdulhameed 0 Jul 26 06:55 pid -> 'pid:[4026531836]'
lrwxrwxrwx 1 abdulhameed abdulhameed 0 Jul 26 07:36 pid_for_children -> 'pid:[4026531836]'
lrwxrwxrwx 1 abdulhameed abdulhameed 0 Jul 26 07:32 time -> 'time:[4026531834]'
lrwxrwxrwx 1 abdulhameed abdulhameed 0 Jul 26 07:36 time_for_children -> 'time:[4026531834]'
lrwxrwxrwx 1 abdulhameed abdulhameed 0 Jul 26 06:55 user -> 'user:[4026531837]'
lrwxrwxrwx 1 abdulhameed abdulhameed 0 Jul 26 06:55 uts -> 'uts:[4026531838]'

#  <type> -> ‘<type>:[<inode>]’. 
# Firstly, <type> is the type of namespace, such as cgroup, ipc, mnt, net, pid, or user. 
# Then, the <inode> is the inode number that uniquely identifies each namespace on the system.
# Additionally, we can use these inode numbers to enter the namespaces of the process using the nsenter command:
$ sudo nsenter --net=/proc/5151/ns/net
```

### procfs ('/proc')
- ``/proc`` is referred to as a virtual file system, so this file system is not used for storage. 
- its main purpose is to provide a file-based interface to hardware, memory, running processes, and other system components.
- real-time information can be retrieved on many system components by viewing the corresponding /proc file.
- 일반적인 파일 시스템은 메모리(/proc가 있는 곳)가 아닌 디스크에 위치하며, 이 경우 인덱스 노드(index-node, 줄여서 inode) 번호는 파일의 inode가 있는 디스크 위치를 가리키는 포인터입니다. 
inode는 파일의 권한과 같은 파일에 관한 정보를 담고 있으며, 파일 데이터가 저장된 디스크 위치를 가리키는 포인터도 포함하고 있다.
- similarly, '/sys' is a virtual file system that provides an interface to kernel data structures. It is commonly mounted at '/sys'.

### Types of Namespaces
- Mount Namespace
  - isolate the set of filesystem mount points seen by a group of processes. 
  thus, processes in different mount namespaces can have different views of the filesystem hierarchy.
- UTS Namespace
- IPC Namespace
- PID Namespace
- Network Namespace
- User Namespace
- Cgroup Namespace
  - each cgroup namespace has its own set of cgroup root directories. when a process creates a new cgroup namespace using clone(2) or unshare(2), its current cgroups directories become the cgroup root directories of the new namespace.
  - creating a different cgroup namespace essentially moves the root directory of the cgroup. If the cgroup was, for example, "/sys/fs/cgroup/mycgroup", a new namespace cgroup could use this as a root directory. 
  the host might see "/sys/fs/cgroup/mycgroup/{group1,group2,group3}" but creating a new cgroup namespace would mean that the new namespace would only see {"group1,group2,group3}".

## Cgroup and Namespace
네임스페이스와 cgroup은 컨테이너 및 최신 애플리케이션을 위한 빌딩 블록입니다. 애플리케이션을 보다 현대적인 아키텍처로 리팩토링할 때 작동 방식을 이해하는 것이 중요합니다.
네임스페이스는 시스템 리소스의 격리를 제공하고 cgroup은 해당 리소스에 대한 세분화된 제어 및 제한 적용을 허용합니다.
컨테이너는 네임스페이스와 cgroup을 사용할 수 있는 유일한 방법이 아닙니다. 네임스페이스 및 cgroup 인터페이스는 Linux 커널에 내장되어 있으므로 다른 애플리케이션에서 이를 사용하여 분리 및 리소스 제약을 제공할 수 있습니다.

## Container File System
### Linux Overlay Filesystem
![blog10_overlay_filesystem.png](images%2Fblog10_overlay_filesystem.png)
- OverlayFS is a union mount filesystem on Linux. It is used to overlay an upper directory tree on top of a lower directory tree (these are virtually merged even if they belong to different filesystems). 
the interesting thing about OverlayFS is that the lower directory tree is read-only, while the upper partition can be modified.

- Modification : any changes made to the files from the upper directory tree will be carried out as usual. however, any changes made to the lower tree are temporary and stored on the view level. 
this means that a copy of the modified files will be created in the upper directory and undergo the changes instead of the original file in the lower layer

- Removal : removing a file from the OverlayFS directory will successfully remove a file from the upper directory, but if the file belongs to the lower directory, OverlayFS will simulate that removal by creating a whiteout file. 
this file will only exist in the OverlayFS directory – no physical changes will be seen in the other two directories. so when OverlayFS is dismounted, this information will be lost.
(Lower Dir 파일에 변경이 필요하면 Upper Dir로 파일을 먼저 복사(Copy) 한 뒤에 변경을 처리합니다(CoW, Copy-On-Write). Upper Dir이 있음으로 해서 기존 이미지 레이어, 즉 원본을 수정하지 않고도 신규 쓰기나 수정이 발생한 변경 부분만 새로 커밋하여 레이어를 추가/확장할 수 있습니다.)
#### exmaple
```bash
/tmp/rootfs# mount -t overlay overlay -o lowerdir=image2:image1,upperdir=container,workdir=work merge
```
![blog10_overlay_filesystem_example.png](images%2Fblog10_overlay_filesystem_example.png)

### Docker Image & Container Layer
#### Image Layer
```dockerfile
# syntax=docker/dockerfile:1

FROM ubuntu:22.04
LABEL org.opencontainers.image.authors="org@example.com"
COPY . /app
RUN make /app
RUN rm -r $HOME/.cache
CMD python /app/app.py
```
- each command modifies the filesystem create a layer.
each layer is only a set of differences from the layer before it. note that both adding, and removing files will result in a new layer. 
in the example above, the ``$HOME/.cache`` directory is removed, but will still be available in the previous layer and add up to the image's total size.
- Image Layer : read-only (=low_dir in OverlayFS)
- Container Layer : read-write (=upper_dir in OverlayFS)
![blog10_docker_layers_1.png](images%2Fblog10_docker_layers_1.png)
#### Container Layer
- the layers are stacked on top of each other. when you create a new container, you add a new writable layer on top of the underlying layers. 
this layer is often called the "container layer". all changes made to the running container, such as writing new files, modifying existing files, and deleting files, are written to this thin writable container layer.
- the major difference between a container and an image is the top writable layer. 
all writes to the container that add new or modify existing data are stored in this writable layer. when the container is deleted, the writable layer is also deleted. The underlying image remains unchanged.
- because each container has its own writable container layer, and all changes are stored in this container layer, multiple containers can share access to the same underlying image and yet have their own data state.
![blog10_docker_layers_2.png](images%2Fblog10_docker_layers_2.png)

#### Implementation
- imageDB : imagedb is an image database that stores information about the Docker layers and the relationships between them (for example, how one depends on the other). the Docker daemon stores the imagedb. 
when running a container, the platform uses this database to retrieve information about each layer.
- layerDB : layerdb is a database that holds information about the relationship between layers. It also holds instructions for building layers. this database is stored in the daemon.
when running a container using an image, the Docker daemon uses both the imagedb and layerdb to start the container.
- caches : cache speeds up the process of building an image. it stores all the files that make up each layer in a database called "layer cache database." 
each file has a hash value and a row in this database and each Docker layer also has a row in the cache database.

#### Where is Docker image and container stored in the host?
- find path liske ``/var/lib/docker``.
- actual image layers are stored in ``/var/lib/docker/image/overlay2``, while the files are stored in ``/var/lib/docker/overlay2``.
- use ``docker inspect`` to find the path of the image and container.
```bash
# docker inspect <container_id>
"GraphDriver": {
            "Data": {
                "LowerDir": "/var/lib/docker/overlay2/756e5194fcdd64be8731377290becc99328245c8865e07c5d51f8aa057253985/diff:/var/lib/docker/overlay2/e780dc911946fcb9e0abbfd0370982f2196efe2d4b15d3c9788216ac1caa4014/diff:/var/lib/docker/overlay2/23ab75ebbe58183a60c984b385a6f02859029f1b6425285ff00238cccd25b7d7/diff:/var/lib/docker/overlay2/ef88c4ee2bb729696b19101c38a19e4be3860509bbbbaf04e3126feaf6248809/diff:/var/lib/docker/overlay2/fc13a95085f3340336bac24b052949c7e1697249b4a60f6e7e3873dbc07a312c/diff:/var/lib/docker/overlay2/e423e037d617a14e5f44606b9460d391eb9d94f9435c160a6d0adb4bfa6fec4b/diff",
                "MergedDir": "/var/lib/docker/overlay2/c3b2470a74361ce94a95738f1aa46ffbe0fa9f782590a7286666a1c9fcaf9ad1/merged",
                "UpperDir": "/var/lib/docker/overlay2/c3b2470a74361ce94a95738f1aa46ffbe0fa9f782590a7286666a1c9fcaf9ad1/diff",
                "WorkDir": "/var/lib/docker/overlay2/c3b2470a74361ce94a95738f1aa46ffbe0fa9f782590a7286666a1c9fcaf9ad1/work"
            },
            "Name": "overlay2"
        },
        "RootFS": {
            "Type": "layers",
            "Layers": [
                "sha256:ceb365432eec83dafc777cac5ee87737b093095035c89dd2eae01970c57b1d15",
                "sha256:84619992a45bb790ab8f77ff523e52fc76dadfe17e205db6a111d0f657d31d71",
                "sha256:3137f8f0c6412c12b46fd397866589505b4474e53580b4e62133da67bf1b2903",
                "sha256:7d52a4114c3602761999a4ea2f84a093c8fcc8662876acc4c3b92878b9948547",
                "sha256:188d128a188cafb013db48e94d9366f0be64083619f50b452cfd093e7affa260",
                "sha256:bcc6856722b7b251ad00728c9cd93b679c7836d5e6780b52316b56c20fd5be94",
                "sha256:61a7fb4dabcd05eba747fed22ff5264f82066d2bf8e16f78198f616e700f5aa7"
            ]
        }
...
# var/lib/docker 
var/lib/docker# tree -L 1
.
├── buildkit
├── containers
├── engine-id
├── image # layers are stored here
├── network
├── overlay2 # files are stored here
├── plugins
├── runtimes
├── swarm
├── tmp
└── volumes

# image layer
/var/lib/docker/image/overlay2# tree -L 2
.
├── distribution
│   ├── diffid-by-digest
│   └── v2metadata-by-diffid
├── imagedb
│   ├── content
│   └── metadata
├── layerdb
│   ├── sha256
│   └── tmp
└── repositories.json

# Layer DB
/var/lib/docker/image/overlay2/layerdb/sha256# tree .
.
├── 0814ebf6e0ed919bf8bf686038d645aa2b535eb9a6bc4b58b2df1b31d499fe3d
│   ├── cache-id # local file hash (~/var/lib/docker/overlay2/~
│   ├── diff # layer file hash
│   ├── parent # parent layer file hash
│   ├── size
│   └── tar-split.json.gz
├── 0baf2321956a506afcddaafe217bc852e4c56a9640530b1b2f98b3378d4b6173
│   ├── cache-id
│   ├── diff
│   ├── size
│   └── tar-split.json.gz
├── 0c2e669c3c8abe5ce516bd0ffbb3dec76614a9cd1dec058a7c4815a403adee83
│   ├── cache-id
│   ├── diff
│   ├── parent
│   ├── size
│   └── tar-split.json.gz
├── 1084f34dba33ee0238270b757d7d4c3ffa06fcac38f1be5bf26bf35d8982eb17
│   ├── cache-id
│   ├── diff
│   ├── parent
│   ├── size
│   └── tar-split.json.gz
...
```

## Container Networking
### Types of Container Network
 - Bridge
 - Host
 - None
 - Overlay
 - Underlay
 - Macvlan

### Docker Network

## Docker in Practice
### Multi-Staged Builds
 - generated images are smaller. the final image is typically much smaller than the one produced by a normal build, as the resulting image includes just what the application needs to run.
 - resulting container will be more secure because your final image includes only what it needs to run the application.
 - smaller images mean less time to transfer or quicker CI/CD builds, faster deployment time, and improved performance.
#### Implement
```dockerfile
# syntax=docker/dockerfile:1

FROM alpine:latest AS builder
RUN apk --no-cache add build-base

FROM builder AS build1
COPY source1.cpp source.cpp
RUN g++ -o /binary source.cpp

FROM builder AS build2
COPY source2.cpp source.cpp
RUN g++ -o /binary source.cpp
```

## Reference
### Cgroup and Namespace
- https://data-flair.training/blogs/system-call-in-os/
- https://www.geeksforgeeks.org/difference-between-user-mode-and-kernel-mode/
- http://man.he.net/man7
- https://www.javatpoint.com/process-vs-thread
- https://navigatorkernel.blogspot.com/
- https://nginxstore.com/blog/kubernetes/%EB%84%A4%EC%9E%84%EC%8A%A4%ED%8E%98%EC%9D%B4%EC%8A%A4%EC%99%80-cgroup%EC%9D%80-%EB%AC%B4%EC%97%87%EC%9D%B4%EB%A9%B0-%EC%96%B4%EB%96%BB%EA%B2%8C-%EC%9E%91%EB%8F%99%ED%95%A9%EB%8B%88%EA%B9%8C/
- https://www.schutzwerk.com/en/blog/linux-container-cgroups-01-intro/
- https://tech.kakaoenterprise.com/154
- https://itnext.io/breaking-down-containers-part-0-system-architecture-37afe0e51770
- https://www.redhat.com/sysadmin/cgroups-part-one
- https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/6/html/resource_management_guide/index
- https://www.baeldung.com/linux/find-process-namespaces
- https://wikidocs.net/196798
- https://lwn.net/Articles/531114/#series_index
- https://www.techtarget.com/searchdatacenter/definition/kernel

### Filesystem
- https://www.educative.io/answers/what-is-overlayfs
- https://tech.kakaoenterprise.com/171
- https://docs.docker.com/storage/storagedriver/
- https://blog.packagecloud.io/what-are-docker-image-layers/

### Network
- https://www.vmware.com/topics/glossary/content/container-networking.html

### Docker In Practice
- https://www.cherryservers.com/blog/docker-multistage-build
- https://docs.docker.com/build/building/multi-stage/