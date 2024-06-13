import sys
import automobile as a

print(sys.argv)

c = a.Motorcycle("Yamaha")
print("Made a bike called: %s" % c.get_name())
c.ride("Mllholland")

b = a.Motorcycle("Zazuki")
print("Made a bike called: %s" % b.get_name())
b.ride("Kissinger")

f = a.Student("M1")
print("Added a student: %s" % f.get_stud_id())
f.start_hpx(sys.argv)
f.do_fut()
print(f.add(10,10))
f.stop_hpx()
