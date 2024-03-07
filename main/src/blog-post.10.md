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

## System Calls
### process vs thread
 - A process is the execution of a program. It includes the program itself, data, resources such as files, and execution info such as process relation information kept by the OS. 
The OS allows users to create, schedule, and terminate the processes via system calls.
 - A thread is a semi-process. It has its own stack and executes a given piece of code. Unlike a real process, the thread normally shares its memory with other threads. 
Conversely, processes usually have a different memory area for each one of them.

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

### procfs


### Types of Namespaces
- Mount Namespace
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

## Container Image and Container Runtime
### Union File System 

## Networking



## Reference
- http://man.he.net/man7
- https://navigatorkernel.blogspot.com/
- https://nginxstore.com/blog/kubernetes/%EB%84%A4%EC%9E%84%EC%8A%A4%ED%8E%98%EC%9D%B4%EC%8A%A4%EC%99%80-cgroup%EC%9D%80-%EB%AC%B4%EC%97%87%EC%9D%B4%EB%A9%B0-%EC%96%B4%EB%96%BB%EA%B2%8C-%EC%9E%91%EB%8F%99%ED%95%A9%EB%8B%88%EA%B9%8C/
- https://www.schutzwerk.com/en/blog/linux-container-cgroups-01-intro/
- https://tech.kakaoenterprise.com/154
- https://itnext.io/breaking-down-containers-part-0-system-architecture-37afe0e51770
- https://www.redhat.com/sysadmin/cgroups-part-one
- https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/6/html/resource_management_guide/index
- https://www.baeldung.com/linux/find-process-namespaces