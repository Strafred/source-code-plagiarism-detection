import java.util.Scanner;
public class inout{
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter the radius and length of a cylinder : ");
        double rad = sc.nextDouble();
        double l = sc.nextDouble();
        double ar = rad*rad*3.14159;
        double vol = ar*l;
        System.out.println("The area is " + ar);
        System.out.println("The volume of the cylinder is " + vol);
    }
}
