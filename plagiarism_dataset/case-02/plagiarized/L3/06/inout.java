import java.util.Scanner;
public class inout{
    public static void main(String[] args) {
        double rad,l,ar,vol;
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter the radius and length of a cylinder : ");
        rad = sc.nextDouble();
        l = sc.nextDouble();
        ar = rad*rad*3.14159;
        vol = ar*l;
        System.out.println("The area is " + ar);
        System.out.println("The volume of the cylinder is " + vol);
    }
}
