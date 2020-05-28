
Shell scripting is a very handy skill to have, especially when you find yourself doing the same thing over and over.
In this lesson, I'll show you some practical examples that will hopefully save you some time.
Please open up a bash shell for these examples.

Example 1: Looping
---

Let's say you have a program that takes one argument, and you would like to run it many times, each with different arguments.
Maybe you want to test how well the program performs depending on the argument.

```bash
$ for i in {0..100}
do
  echo "Running with argument $i"
  ./my_program $i
done
Running with argument 0
Running with argument 1
Running with argument 2
Running with argument 3
...
```

You can also use another syntax which may feel more familiar:

```bash
$ for ((i=0; i<100; i++))
do
  echo "Running with argument $i"
  ./my_program $i
done
Running with argument 0
Running with argument 1
Running with argument 2
Running with argument 3
...
```

File Processing
---

File processing is probably the most comman task shell scripting is used for, and the task for which it is best suited.
Let's say for example, we have a file which contains the labels for a text dataset like so:
```bash
$ cat dataset.txt
positive One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked. The...
positive A wonderful little production. <br /><br />The filming technique is very unassuming- very old-time-B...
positive I thought this was a wonderful way to spend time on a too hot summer weekend, sitting in the air con...
negative Basically there's a family where a little boy (Jake) thinks there's a zombie in his closet & his par...
positive Petter Mattei's "Love in the Time of Money" is a visually stunning film to watch. Mr. Mattei offers ...
positive Probably my all-time favorite movie, a story of selflessness, sacrifice and dedication to a noble ca...
positive I sure would like to see a resurrection of a up dated Seahunt series with the tech they have today i...
negative This show was an amazing, fresh & innovative idea in the 70's when it first aired. The first 7 or 8 ...
negative Encouraged by the positive comments about this film on here I was looking forward to watching this f...
positive If you like original gut wrenching laughter you will like this movie. If you are young or old then y...
negative Phil the Alien is one of those quirky films where the humour is based around the oddness of everythi...
...
```

This dataset may be very large. Let's say that we only wanted to look at the first 5 lines of the dataset.
The `head` command is perfect for this.
You may pass an argument to specify the number of lines: `head -n 13` for the first 13 lines.
Tail works in the same way, but looks at the end of the file.

```bash
$ head -n 3 dataset.txt
positive One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked. The...
positive A wonderful little production. <br /><br />The filming technique is very unassuming- very old-time-B...
positive I thought this was a wonderful way to spend time on a too hot summer weekend, sitting in the air con...
```

Perhaps we only want to look at the positive reviews.
`grep` is well suited for this.
Grep searches for a regular expression in a given file or text stream.

```bash
$ grep 'positive' dataset.txt
positive One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked. The...
positive A wonderful little production. <br /><br />The filming technique is very unassuming- very old-time-B...
positive I thought this was a wonderful way to spend time on a too hot summer weekend, sitting in the air con...
positive Petter Mattei's "Love in the Time of Money" is a visually stunning film to watch. Mr. Mattei offers ...
positive Probably my all-time favorite movie, a story of selflessness, sacrifice and dedication to a noble ca...
positive I sure would like to see a resurrection of a up dated Seahunt series with the tech they have today i...
...
```

Perhaps we only want to look at the first 3 lines which contain positive reviews.
How might we combine the `head` and `grep` commands?
For this we will use the `|`, or the pipe character.
`|` makes the output of one command the input of the next, like so:

```bash
$ grep 'positive' dataset.txt | head -n 3
positive One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked. The...
positive A wonderful little production. <br /><br />The filming technique is very unassuming- very old-time-B...
positive I thought this was a wonderful way to spend time on a too hot summer weekend, sitting in the air con...
```

```bash
```

```bash
```
